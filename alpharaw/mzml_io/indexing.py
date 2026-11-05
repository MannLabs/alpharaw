"""Turn a plain mzML byte stream into a spec-compliant ``indexedmzML`` document.

The ``indexedmzML`` wrapper carries the byte offset of every ``<spectrum>`` element,
letting a reader seek directly to a spectrum instead of parsing the whole document.
Several downstream tools require it; DIA-NN, for instance, refuses plain mzML outright.

The index is *appended*. Because the ``<indexedmzML>`` header is written before any
spectrum, no byte position shifts when the index is added, so the document never has to
be rewritten or copied through a temporary file.

Offsets come from expat's ``CurrentByteIndex``, which reports the exact byte position of
each start tag. Deriving them from the serialized XML rather than from the element tree
keeps them correct regardless of indentation or formatting choices made by the writer.
"""

from __future__ import annotations

import hashlib
import xml.parsers.expat
from pathlib import Path

from alpharaw.mzml_io.cv_constants import XML

__all__ = [
    "INDEXED_MZML_FOOTER_TAG",
    "INDEXED_MZML_HEADER",
    "append_spectrum_index",
]

#: Schema declared by an indexed document; differs from the plain-mzML schema.
_SCHEMA_LOCATION_IDX = (
    f"{XML.NS_URI_MZML} http://psidev.info/files/ms/mzML/xsd/mzML1.1.2_idx.xsd"
)

#: Opening bytes of an indexed document, written before any spectrum.
INDEXED_MZML_HEADER = (
    b'<?xml version="1.0" encoding="utf-8"?>\n'
    b'<indexedmzML xmlns="'
    + XML.NS_URI_MZML.encode()
    + b'" xmlns:xsi="'
    + XML.NS_URI_XSI.encode()
    + b'" xsi:schemaLocation="'
    + _SCHEMA_LOCATION_IDX.encode()
    + b'">\n'
)

INDEXED_MZML_FOOTER_TAG = b"</indexedmzML>\n"

_SPECTRUM_TAG = "spectrum"
_READ_CHUNK_SIZE = 1 << 20  # 1 MiB


def _scan_offsets_and_digest(path: Path) -> tuple[list[tuple[str, int]], hashlib._Hash]:
    """Collect spectrum byte offsets and hash the bytes in a single read pass.

    The document on disk is still unterminated at this point: ``<indexedmzML>`` has no
    closing tag yet. The parser is therefore never finalized (``isfinal`` stays False),
    which lets expat parse the incomplete prefix without raising.

    Parameters
    ----------
    path : Path
        File holding the header plus the ``<mzML>`` element.

    Returns
    -------
    tuple
        ``(offsets, hasher)`` where ``offsets`` is a list of ``(spectrum id, byte
        offset)`` in document order, and ``hasher`` has consumed every byte read.

    """
    offsets: list[tuple[str, int]] = []
    parser = xml.parsers.expat.ParserCreate()

    def _on_start_element(name: str, attrs: dict[str, str]) -> None:
        if name == _SPECTRUM_TAG:
            # CurrentByteIndex points at the '<' of this start tag.
            offsets.append((attrs["id"], parser.CurrentByteIndex))

    parser.StartElementHandler = _on_start_element

    hasher = hashlib.sha1()  # noqa: S324 - mandated by the mzML specification
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_SIZE):
            hasher.update(chunk)
            # isfinal stays False: the document is still unterminated at this point.
            parser.Parse(chunk)

    return offsets, hasher


def _format_index(offsets: list[tuple[str, int]]) -> bytes:
    """Render the ``<indexList>`` block for the given spectrum offsets."""
    lines = [b'<indexList count="1">\n', b'  <index name="spectrum">\n']
    lines += [
        b'    <offset idRef="%s">%d</offset>\n' % (spec_id.encode(), offset)
        for spec_id, offset in offsets
    ]
    lines += [b"  </index>\n", b"</indexList>\n"]
    return b"".join(lines)


def append_spectrum_index(mzml_path: str | Path) -> int:
    """Append ``indexList``, ``indexListOffset`` and ``fileChecksum`` to a document.

    The file must already hold :data:`INDEXED_MZML_HEADER` followed by a complete
    ``<mzML>`` element, and must not yet be closed with ``</indexedmzML>``.

    Per the mzML specification the checksum is the SHA-1 of the document from its first
    byte up to and including the opening ``<fileChecksum>`` tag. It is accumulated as the
    file is scanned and then extended with the bytes written here, so no additional read
    pass is needed.

    Parameters
    ----------
    mzml_path : str or Path
        Path to the unterminated indexed mzML document, modified in place.

    Returns
    -------
    int
        Number of spectra recorded in the index.

    """
    path = Path(mzml_path)
    offsets, hasher = _scan_offsets_and_digest(path)

    index_offset = path.stat().st_size
    index_block = _format_index(offsets)
    offset_block = b"<indexListOffset>%d</indexListOffset>\n" % index_offset
    checksum_open = b"<fileChecksum>"

    hasher.update(index_block)
    hasher.update(offset_block)
    hasher.update(checksum_open)

    with path.open("ab") as handle:
        handle.write(index_block)
        handle.write(offset_block)
        handle.write(checksum_open)
        handle.write(hasher.hexdigest().encode())
        handle.write(b"</fileChecksum>\n")
        handle.write(INDEXED_MZML_FOOTER_TAG)

    return len(offsets)
