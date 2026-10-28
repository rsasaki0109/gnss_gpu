import struct

import pytest

from experiments.import_boreas_gnss import primary_stream, receiver_fixes, status_intervals


def sentence(body):
    checksum = 0
    for value in body.encode():
        checksum ^= value
    return f"${body}*{checksum:02X}\r\n".encode()


def frame(payload):
    body = bytes(26)+struct.pack("<HIH", 17, 0xffffffff, len(payload))+payload
    body += bytes((-(8+len(body)+4)) % 4)
    packet = bytearray(b"$GRP"+struct.pack("<HH", 10001, len(body)+4)+body+b"\x00\x00$#")
    checksum = -sum(struct.unpack("<"+"H"*(len(packet)//2), packet)) & 65535
    struct.pack_into("<H", packet, len(packet)-4, checksum)
    return bytes(packet)


def test_binary_framing_preserves_split_native_messages_and_embedded_markers():
    data = b"native$GRPbytes"
    payload, counts = primary_stream(frame(data[:7])+frame(data[7:]))
    assert payload == data
    assert counts == {"$GRP:10001": 2}


def test_corrupted_outer_checksum_and_truncated_frame_are_rejected():
    packet = bytearray(frame(b"receiver data")); packet[43] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        primary_stream(packet)
    with pytest.raises(ValueError, match="truncated"):
        primary_stream(frame(b"receiver data")[:-1])


def test_zda_date_carry_at_midnight_and_ellipsoid_height():
    data = sentence("GNGGA,235959.00,4300.0,N,07900.0,W,1,11,0.9,196.0,M,-37.0,M,,")
    data += sentence("GNZDA,235959.00,26,11,2020,00,00")
    data += sentence("GNGGA,000000.00,4300.0,N,07900.0,W,4,08,1.5,196.0,M,-37.0,M,,")
    rows = receiver_fixes(data)
    assert rows[0]["longitude_deg"] == -79
    assert rows[0]["ellipsoid_height_m"] == 159
    assert rows[1]["time_utc"] == "2020-11-27T00:00:00+00:00"


def test_no_fix_epochs_are_retained_in_availability_denominator():
    data = sentence("GNZDA,120000.00,26,11,2020,00,00")
    data += sentence("GNGGA,120000.00,,,,,0,00,,,,,,,")
    rows = receiver_fixes(data)
    assert len(rows) == 1
    assert not rows[0]["receiver_reports_fix"]
    assert rows[0]["latitude_deg"] is None


def test_bad_nmea_checksum_and_missing_date_are_not_accepted():
    data = sentence("GNZDA,120000.00,26,11,2020,00,00")
    broken = bytearray(data); broken[7] ^= 1
    with pytest.raises(ValueError, match="NMEA checksum"):
        receiver_fixes(broken)
    with pytest.raises(ValueError, match="ZDA date required"):
        receiver_fixes(sentence("GNGGA,120000.00,,,,,0,00,,,,,,,"))


def test_stale_date_anchor_and_duplicate_epochs_are_rejected():
    zda = sentence("GNZDA,120000.00,26,11,2020,00,00")
    gga = sentence("GNGGA,120000.00,,,,,0,00,,,,,,,")
    with pytest.raises(ValueError, match="duplicate"):
        receiver_fixes(zda+gga+gga)
    with pytest.raises(ValueError, match="too far"):
        receiver_fixes(zda+sentence("GNGGA,120100.00,,,,,0,00,,,,,,,"))


def test_status_intervals_split_on_gap_even_when_quality_is_unchanged():
    rows = [dict(time_utc=f"2020-11-26T12:00:{second:02d}+00:00", quality_code=4,
                 satellites_used=8, hdop=1.2) for second in (0, 1, 4)]
    intervals = status_intervals(rows)
    assert [r["observed_epochs"] for r in intervals] == [2, 1]
    assert [r["observed_span_s"] for r in intervals] == [1., 0.]


def test_status_change_and_no_fix_are_retained_without_assumed_duration():
    rows = [dict(time_utc=f"2020-11-26T12:00:0{index}+00:00", quality_code=quality,
                 satellites_used=None, hdop=None) for index, quality in enumerate((4, 0, 4))]
    intervals = status_intervals(rows)
    assert [r["quality_code"] for r in intervals] == [4, 0, 4]
    assert all(r["observed_span_s"] == 0. for r in intervals)
