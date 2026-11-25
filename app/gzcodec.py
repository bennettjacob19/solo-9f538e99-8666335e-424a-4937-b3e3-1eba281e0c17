"""Standard-gzip helpers: validation, multi-member rebuild, sparse index, range reads.

A "multi-member" gzip file is a concatenation of independent gzip members. This is
explicitly allowed by RFC 1952 (a gzip file consists of a series of "members"), and
the system `gzip -dc` transparently decompresses the concatenation. Each member here
holds at most MEMBER_SIZE bytes of uncompressed data, so a random range can be served
by decompressing only the members overlapping it. A request of at most one member's
worth of bytes (64 KiB) can span at most 2 members, i.e. at most 128 KiB of actual
decompression, comfortably below the 256 KiB requirement (a hard guard enforces it).

The index only stores (uncompressed offset/size, compressed offset/size) tuples per
member. It is never a copy of the decompressed log body.
"""

from __future__ import annotations

import bisect
import json
import zlib

GZIP_MAGIC = b"\x1f\x8b"
MEMBER_SIZE = 64 * 1024
MAX_DECOMPRESS_PER_RANGE = 256 * 1024
GZIP_WBITS = 31  # zlib magic for raw gzip streams


class GzipError(ValueError):
    """Raised when input is not a valid standard gzip stream."""


class QuotaExceeded(RuntimeError):
    """New-file byte cap would be exceeded."""


class SimulatedWriteFailure(RuntimeError):
    """Fault injection: simulated write failure."""


def iter_decompressed(fp, buf_size: int = 65536):
    """Yield decompressed chunks from a (possibly multi-member) standard gzip file.

    A gzip member can end exactly at a read() boundary: the final bytes of one
    member may arrive in a chunk of exactly ``buf_size`` bytes, while the next
    member only appears on the following read(). In that case ``unused_data`` is
    still empty when EOF is first observed, so the loop must issue one more read
    before concluding the stream is finished -- otherwise every member after a
    boundary-aligned member would silently disappear (the classic "the web view
    misses the tail of the log but the original download is complete" bug).
    """
    pending = b""  # bytes read but not yet fed to a decompressobj
    d = zlib.decompressobj(GZIP_WBITS)
    fed = False
    while True:
        if not pending:
            pending = fp.read(buf_size)
        if pending:
            out = d.decompress(pending)
            pending = b""
            fed = True
            if out:
                yield out
        if d.eof:
            tail = d.flush()
            if tail:
                yield tail
            pending = d.unused_data  # start of the next member, if any
            if pending:
                # Next member's bytes already rode in on the same read chunk.
                d = zlib.decompressobj(GZIP_WBITS)
                fed = False
                continue
            # The member ended at (or before) the end of the last read chunk.
            # One more read decides whether another member follows.
            pending = fp.read(buf_size)
            if pending:
                d = zlib.decompressobj(GZIP_WBITS)
                fed = False
                continue
            return
        if not pending:
            # Input exhausted while the current member is unfinished.
            if not fed:
                raise GzipError("empty or non-gzip input")
            raise GzipError("truncated gzip stream")


def validate_gzip(path: str) -> int:
    """Validate a standard gzip file and return its total uncompressed size."""
    with open(path, "rb") as fp:
        magic = fp.read(2)
        if magic != GZIP_MAGIC:
            raise GzipError("not a gzip file (bad magic bytes)")
        fp.seek(0)
        total = 0
        for chunk in iter_decompressed(fp):
            total += len(chunk)
    return total


def rebuild_multimember(src_path: str, write, member_size: int = MEMBER_SIZE):
    """Decompress src gzip and re-emit it as concatenated <=member_size members.

    `write(bytes)` receives every compressed output byte (gzip and, via callers,
    index bytes) so callers can enforce byte caps and measure peaks.

    Returns (index_dict, uncompressed_size). The index contains one entry per
    member with compressed/uncompressed offsets and sizes only.
    """
    members = []
    coff = 0
    uoff = 0
    total = 0
    member_used = 0
    member_start_written = 0
    cobj = zlib.compressobj(6, zlib.DEFLATED, GZIP_WBITS)

    def close_member():
        nonlocal coff, uoff, cobj, member_used, member_start_written
        tail = cobj.flush()
        if tail:
            write(tail)
        csize = account["bytes"] - member_start_written
        # A member entry is always appended (including the zero-byte final member
        # of an empty log) so the index table is never empty.
        members.append({
            "uoff": uoff,
            "usize": member_used,
            "coff": coff,
            "csize": csize,
        })
        coff += csize
        uoff += member_used
        cobj = zlib.compressobj(6, zlib.DEFLATED, GZIP_WBITS)
        member_used = 0
        member_start_written = account["bytes"]

    # Track compressed bytes inside the gzip file only (not index bytes).
    account = {"bytes": 0}
    raw_write = write

    def gzip_write(data):
        raw_write(data)
        account["bytes"] += len(data)

    write = gzip_write

    with open(src_path, "rb") as src:
        pending = b""
        for chunk in iter_decompressed(src):
            total += len(chunk)
            pending += chunk
            while pending:
                room = member_size - member_used
                take = pending[:room]
                pending = pending[room:]
                if take:
                    out = cobj.compress(take)
                    if out:
                        write(out)
                    member_used += len(take)
                if member_used == member_size:
                    close_member()
        close_member()  # final (possibly empty) member

    index = {
        "version": 1,
        "member_size": member_size,
        "uncompressed_size": total,
        "members": members,
    }
    return index, total


def rebuild_tail_members(src_path: str, skip_uncompressed: int, write,
                         member_size: int = MEMBER_SIZE):
    """Recompress only the uncompressed tail after ``skip_uncompressed`` bytes.

    Used by the legacy-log repair: the already readable prefix (and the fast
    artifacts covering it) is left untouched, so new disk usage grows only with
    the missing tail. Emits concatenated independent gzip members of at most
    ``member_size`` uncompressed bytes each, exactly like rebuild_multimember.

    Returns (members, tail_size, total_size) where members is an index-table
    fragment (same uoff/usize/coff/csize fields; offsets are absolute, compressed
    offsets start at zero relative to the staging file), tail_size is the number
    of uncompressed bytes the fragment covers, and total_size is the true total
    uncompressed size of the source gzip.
    """
    members = []
    coff = 0
    uoff = skip_uncompressed
    member_used = 0
    total = 0
    member_start_written = 0
    cobj = zlib.compressobj(6, zlib.DEFLATED, GZIP_WBITS)
    account = {"bytes": 0}
    raw_write = write

    def gzip_write(data):
        raw_write(data)
        account["bytes"] += len(data)

    write = gzip_write

    def close_member():
        nonlocal coff, uoff, cobj, member_used, member_start_written
        tail = cobj.flush()
        if tail:
            write(tail)
        csize = account["bytes"] - member_start_written
        members.append({
            "uoff": uoff,
            "usize": member_used,
            "coff": coff,
            "csize": csize,
        })
        coff += csize
        uoff += member_used
        cobj = zlib.compressobj(6, zlib.DEFLATED, GZIP_WBITS)
        member_used = 0
        member_start_written = account["bytes"]

    with open(src_path, "rb") as src:
        remaining_skip = skip_uncompressed
        pending = b""
        for chunk in iter_decompressed(src):
            total += len(chunk)
            if remaining_skip > 0:
                if len(chunk) <= remaining_skip:
                    remaining_skip -= len(chunk)
                    continue
                chunk = chunk[remaining_skip:]
                remaining_skip = 0
            pending += chunk
            while pending:
                room = member_size - member_used
                take = pending[:room]
                pending = pending[room:]
                if take:
                    out = cobj.compress(take)
                    if out:
                        write(out)
                    member_used += len(take)
                if member_used == member_size:
                    close_member()
        # Flush the final member whenever the tail is non-empty.
        if member_used or members:
            close_member()

    tail_size = total - skip_uncompressed
    if tail_size < 0:
        raise GzipError(
            f"source uncompressed size {total} is smaller than the registered "
            f"readable size {skip_uncompressed}; refusing repair")
    return members, tail_size, total


def read_index(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        index = json.load(fp)
    if not isinstance(index.get("members"), list) or not index["members"]:
        raise GzipError("index is missing member table")
    return index


def read_range_fast(gzip_path: str, index: dict, start: int, length: int):
    """Read [start, start+length) from a multi-member gzip using its index.

    Returns (data, decompressed_bytes). decompressed_bytes counts every byte
    actually inflated while serving the request and is guaranteed <= 128 KiB for
    a request of at most 64 KiB with 64 KiB members (at most 2 members touched);
    a hard 256 KiB guard backs the guarantee.
    """
    size = index["uncompressed_size"]
    if start < 0 or start >= size:
        raise ValueError("start out of range")
    end = min(start + length, size)
    members = index["members"]
    offsets = [m["uoff"] for m in members]
    i = bisect.bisect_right(offsets, start) - 1
    if i < 0:
        raise GzipError("index does not cover requested offset")

    out = []
    decompressed = 0
    cursor = start
    with open(gzip_path, "rb") as fp:
        while cursor < end:
            if i >= len(members):
                raise GzipError("index ends before requested range")
            m = members[i]
            fp.seek(m["coff"])
            raw = fp.read(m["csize"])
            if len(raw) != m["csize"]:
                raise GzipError("compressed member is shorter than indexed size")
            d = zlib.decompressobj(GZIP_WBITS)
            data = d.decompress(raw) + d.flush()
            if len(data) != m["usize"]:
                raise GzipError("member size does not match index")
            decompressed += len(data)
            if decompressed > MAX_DECOMPRESS_PER_RANGE:
                raise RuntimeError(
                    f"range read decompressed {decompressed} bytes, "
                    f"limit is {MAX_DECOMPRESS_PER_RANGE}"
                )
            local_start = cursor - m["uoff"]
            local_end = min(end, m["uoff"] + len(data)) - m["uoff"]
            out.append(data[local_start:local_end])
            cursor = m["uoff"] + len(data)
            i += 1
    return b"".join(out), decompressed


def read_range_basic(gzip_path: str, start: int, length: int):
    """Slow path used before (and during) an upgrade: full decompression."""
    chunks = []
    decompressed = 0
    with open(gzip_path, "rb") as fp:
        for chunk in iter_decompressed(fp):
            chunks.append(chunk)
            decompressed += len(chunk)
    return b"".join(chunks)[start:start + length], decompressed
