# GSDC 2023 DGNSS feasibility

調査日: 2026-04-17

## 結論

GSDC 2023 に NOAA CORS base station を使った DGNSS / DD pseudorange を入れるのは feasible。
ただし、既存 `python/gnss_gpu/dd_pseudorange.py` をそのまま GSDC に接続するだけでは足りない。

必要な追加作業:
- GSDC `device_gnss.csv` row を `DDPseudorangeComputer.compute_dd()` の rover measurement 形式へ変換する adapter。
- NOAA CORS RINEX 2.11 / Hatanaka (`.d.gz`) を読める取得・変換層。
- 1 Hz rover epoch に対して、base 観測も 1 Hz で取得する経路。30 s daily file だけでは不十分。
- station/date ごとの fallback。近い station が全期間で常に揃うわけではない。

次セッションで prototype 実装に進める。見積は 2-4 engineer-days。

## 参照ソース

- NOAA NCN API: https://geodesy.noaa.gov/web_services/ncn-api.shtml
- NOAA NCN data FAQ: https://geodesy.noaa.gov/CORS/cors_faqs.shtml
- NOAA CORS data root/readme: https://geodesy.noaa.gov/corsdata/ and https://geodesy.noaa.gov/corsdata/readme.txt
- NCEI CORS archive: https://www.ncei.noaa.gov/products/continuously-operating-reference-stations
- NASA CDDIS high-rate 1 s GNSS dataset: https://data.nasa.gov/dataset/cddis-gnss-highrate-data-a052f
- IGS data archive notes: https://igs.org/data/

## Candidate stations

NOAA NCN API の nearest NCN stations service で、代表点を WGS84 -> ECEF に変換して照会した。
距離は API response の直線距離。

Nearest API queries:
- Mountain View: https://geodesy.noaa.gov/api/nde/ncors?x=-2695156.445&y=-4299130.640&z=3851527.438
- San Jose: https://geodesy.noaa.gov/api/nde/ncors?x=-2682017.877&y=-4311139.317&z=3847302.075
- LAX: https://geodesy.noaa.gov/api/nde/ncors?x=-2519955.552&y=-4658985.901&z=3541157.167
- Los Angeles center: https://geodesy.noaa.gov/api/nde/ncors?x=-2503357.312&y=-4660203.434&z=3551245.359

| GSDC area | target point | nearest candidates | recommended use |
|---|---|---|---|
| Mountain View / Palo Alto | `37.3861, -122.0839` | `SLAC` 11.18 km, `P222` 17.00 km, `ZOA2/ZOA1` 18.43 km | `SLAC` primary, `P222` fallback |
| San Jose | `37.3382, -121.8863` | `MHC2/MHCB` 21.60 km, `ZOA2/ZOA1` 25.46 km, `P222` 28.31 km, `SLAC` 29.47 km, `P217` 33.32 km | date-aware: `MHC2`/`MHCB` when available; robust fallback `P222` or `P217` |
| LAX routes | `33.9425, -118.4081` | `TORP` 17.58 km, `CRHS` 18.19 km, `PVEP` 22.10 km, `VDCY` 31.42 km, `JPLM` 36.29 km | `TORP` primary, `CRHS` fallback; `VDCY/JPLM` for north/east LA segments |
| Los Angeles center | `34.0522, -118.2437` | `VDCY` 14.19 km, `JPLM` 18.13 km, `CRHS` 25.51 km, `TORP` 29.34 km | `VDCY` primary, `JPLM` fallback |

Station pages checked:
- `SLAC`: https://www.ngs.noaa.gov/CORS/Sites/slac.html
- `P222`: https://www.ngs.noaa.gov/CORS/Sites/p222.html
- `MHC2`: https://www.ngs.noaa.gov/CORS/Sites/mhc2.html
- `P217`: https://www.ngs.noaa.gov/CORS/Sites/p217.html
- `TORP`: https://www.ngs.noaa.gov/CORS/Sites/torp.html
- `CRHS`: https://www.ngs.noaa.gov/CORS/Sites/crhs.html
- `VDCY`: https://www.ngs.noaa.gov/CORS/Sites/vdcy.html
- `JPLM`: https://www.ngs.noaa.gov/CORS/Sites/jplm.html

## RINEX access

NOAA FAQ states NCN data are available from the NOAA-NCN AWS bucket, NGS server, UFCORS, and NCEI CLASS.
The direct file pattern is:

```text
https://geodesy.noaa.gov/corsdata/rinex/YYYY/DDD/ssss/ssssDDD0.YYd.gz
https://noaa-cors-pds.s3.amazonaws.com/rinex/YYYY/DDD/ssss/ssssDDD0.YYd.gz
```

Spot checks against GSDC dates:

| station | date | result |
|---|---|---|
| `SLAC` | 2020-06-25 | `slac1770.20d.gz` OK on NGS and AWS |
| `SLAC` | 2023-05-23 | `slac1430.23d.gz` OK; `slac1430.23o.gz` also OK |
| `MHC2` | 2020-12-10 | missing |
| `MHC2` | 2023-06-06 | `mhc21570.23d.gz` OK |
| `MHCB` | 2020-12-10 | `mhcb3450.20d.gz` OK |
| `MHCB` | 2023-05-25 | missing |
| `P222` | 2020-12-10, 2022-02-08, 2023-05-25, 2023-06-06 | OK on all checked dates |
| `P217` | 2020-12-10, 2022-02-08, 2023-05-25, 2023-06-06 | OK on all checked dates |
| `TORP` | 2021-12-07, 2022-02-23 | OK |
| `VDCY` | 2021-12-07 | OK |
| `JPLM` | 2021-12-08 | OK |

Example verified URL:

```text
https://geodesy.noaa.gov/corsdata/rinex/2023/143/slac/slac1430.23d.gz
```

Important detail: the public daily file can be 30 s. For example `slac1430.23o.gz` has RINEX 2.11
header `INTERVAL 30.0000`. GSDC rover epochs are effectively 1 Hz. For DD pseudorange this means:
- 30 s daily files are useful for station availability and first smoke tests.
- Production-quality DGNSS should use UFCORS / NCEI CLASS at-sampling-rate data, or CDDIS high-rate
  1 s files where the station exists there.
- If only 30 s base data are used, apply DD updates only on matching epochs or derive/interpolate
  base corrections carefully after removing geometry. Do not interpolate raw pseudorange blindly.

CDDIS is useful for high-rate 1 s products, but direct CDDIS paths were not verified for the selected
NOAA-NCN candidates in this pass. Prefer NOAA NGS/AWS for deterministic station coverage, then add
CDDIS high-rate as an optional source when station naming and Earthdata access are settled.

## Compatibility with `dd_pseudorange.py`

Existing `DDPseudorangeComputer` expects:
- base RINEX observation file readable by `gnss_gpu.io.rinex.read_rinex_obs`;
- rover rows exposing `corrected_pseudorange`, `satellite_ecef`, `elevation`, `snr`, `system_id`, `prn`;
- GPS time-of-week in seconds;
- common satellite IDs per constellation, then one reference satellite per system.

GSDC has the needed physical fields:
- time: `ArrivalTimeNanosSinceGpsEpoch`;
- satellite ID: `ConstellationType`, `Svid`, `SignalType`;
- corrected pseudorange inputs: `RawPseudorangeMeters`, `SvClockBiasMeters`,
  `IonosphericDelayMeters`, `TroposphericDelayMeters`, `IsrbMeters`;
- satellite position/elevation/CN0: `SvPosition*EcefMeters`, `SvElevationDegrees`, `Cn0DbHz`;
- seed position: `WlsPosition*EcefMeters`.

Adapter mapping:
- `tow = (ArrivalTimeNanosSinceGpsEpoch * 1e-9) % 604800.0`;
- `system_id`: derive from `SignalType`, not raw Android `ConstellationType`. Repo convention is
  `0 -> G`, `1 -> R`, `2 -> E`, `3 -> C`, `4 -> J`; Android IDs are different.
  For the first prototype, map `GPS_* -> 0` and `GAL_* -> 2`.
- `prn = Svid`;
- `corrected_pseudorange = RawPseudorangeMeters + SvClockBiasMeters - IonosphericDelayMeters - TroposphericDelayMeters - IsrbMeters`;
- `snr = Cn0DbHz`;
- `satellite_ecef = [SvPositionX/Y/Z]`.

Signal scope:
- First prototype should use GPS L1 C/A and Galileo E1 only.
- Current `DDPseudorangeComputer` selects C1/E1-style pseudorange codes and intentionally chooses one
  row per satellite. This is compatible with L1/E1, not with mixing L5/E5a.
- Extending to L5/E5a requires explicit C5/C7 code selection and per-frequency grouping. Do this only
  after L1/E1 DD proves useful.

Parser gap:
- `python/gnss_gpu/io/rinex.py` is a RINEX 3-style plain text parser and looks for `>` epoch headers.
- NOAA files encountered here are often RINEX 2.11 and Hatanaka/gzip compressed.
- Implementation should either add `georinex`/external `crx2rnx` + RINEX2 support, or create a small
  RINEX2 observation parser for the C/P code columns needed by DD pseudorange.

## Proposed implementation plan

1. Add a CORS resolver/downloader:
   - map route area (`mtv`, `sjc`, `lax`, generic LA) and date to ordered station candidates;
   - try NOAA AWS, then NGS server;
   - cache files under `/tmp/gsdc_cors/YYYY/DDD/station/`.
2. Add RINEX normalization:
   - `.gz` decompress;
   - `.d` Hatanaka convert to `.o` or parse via a library;
   - support RINEX 2.11 observation headers and epochs.
3. Add GSDC rover adapter:
   - group by `ArrivalTimeNanosSinceGpsEpoch`;
   - choose L1/E1 rows;
   - create lightweight measurement objects for `compute_dd`.
4. Add train experiment:
   - start with WLS + DD pseudorange residual diagnostics, not PF;
   - then attach to PF observation update if residuals are sane;
   - report per-station coverage, DD pair count, residual median, and WLS/PF metric deltas.

## Risks

- Sampling rate is the largest risk. 30 s base data will not support every 1 Hz rover epoch without
  a correction interpolation design.
- Baselines are 10-35 km for the practical candidates. Code DD atmospheric cancellation is helpful
  but not as strong as a local base within a few km.
- Station availability is date-dependent. San Jose needs fallback because the closest Mount Hamilton
  station pair is not present across all checked dates.
- Datum mismatch must be handled explicitly. NOAA API coordinates are NAD 83(2011) epoch 2010.00;
  RINEX headers may contain approximate positions, and DGNSS code should use authoritative station
  coordinates from NOAA coordinate/API output.
- Existing DD gating from UrbanNav should not be copied blindly. Smartphone code multipath and
  inter-signal biases need GSDC-specific residual diagnostics.

## Implementation smoke note

2026-04-17 に first implementation smoke を実施した。詳細は
`internal_docs/experiments.md` の "GSDC 2023 F: NOAA CORS DD pseudorange smoke" を参照。

要点:
- NOAA CORS `.d.gz` 取得、Hatanaka -> RINEX 2 obs 変換、GSDC L1/E1 rover adapter、
  bounded DD WLS update は実装済み。
- CORS RINEX は raw pseudorange なので、GSDC DD 側も raw pseudorange を使う必要がある。
  GSDC satellite-clock corrected pseudorange と CORS raw を混ぜると DD residual が 16 万 m 級になる。
- public daily CORS 30 s file の nearest-epoch smoke では coverage が 3-7% 程度しかなく、
  MTV/LAX smoke とも Android WLS に対して同等または悪化した。
- この条件では PF-100K mean P50 `2.83 m` を下回る見込みがないため、full train と submission 生成には進めない。
