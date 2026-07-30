# Mono-free Thermo reader — Linux test container

This container proves that alpharaw can read Thermo `.raw` files on Linux using
only the cross-platform **.NET 8** runtime (through pythonnet's `coreclr` backend),
with **no Mono** installed.

It relies on the .NET 8 build of Thermo's `RawFileReader` DLLs
(`ThermoFisher.CommonCore.Data.dll` and `ThermoFisher.CommonCore.RawFileReader.dll`,
v8.0.37) bundled in `alpharaw/ext/thermo_fisher/net8/`, sourced from
<https://github.com/thermofisherlsms/RawFileReader> (`Libs/NetCore/Net8`). The
`.../netfx/` folder holds the .NET Framework build used by the `mono` fallback.

## Build & run

From the repository root:

```
docker build -f docker/Dockerfile -t alpharaw-thermo-monofree .
docker run --rm alpharaw-thermo-monofree
```

Expected output ends with:

```
Platform: Linux x86_64
Active .NET runtime: coreclr
spectra: 3937
peaks:   1103989
OK: read .../nbs_tests/test_data/iRT.raw via 'coreclr' runtime
```

(`Platform` reflects the build/host architecture.)

### ARM (aarch64) proof

`Dockerfile.arm64` pins `--platform=linux/arm64` and asserts the container is
running on `aarch64` before reading, so the run is an unambiguous ARM proof:

```
docker build --platform linux/arm64 -f docker/Dockerfile.arm64 -t alpharaw-thermo-arm64 .
docker run --rm alpharaw-thermo-arm64
```

Expected output includes `container arch: aarch64` and `Platform: Linux aarch64`.
This builds natively on Apple Silicon. On an x86-64 host, enable emulation first:

```
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

## Notes

- The runtime backend is selected by `clr_utils.py`: when `ALPHARAW_DOTNET_RUNTIME`
  is unset it prefers a .NET Framework runtime (Mono/netfx) and falls back to
  `coreclr`. These containers set `ALPHARAW_DOTNET_RUNTIME=coreclr` explicitly to
  force the mono-free path.
- `libicu` is installed because the reader sets `CultureInfo("en-US")`; .NET's
  invariant-globalization mode would make that call fail.
- The image builds for the host's architecture by default. Add
  `--platform linux/amd64` to `docker build` to force x86-64.
- The Sciex `.wiff` reader is **not** covered here: its `Clearcore2.Data.dll` still
  targets .NET Framework 4.0 and requires Mono.
