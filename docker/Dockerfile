# Mono-free Thermo reading test for alpharaw on Linux.
#
# Proves that Thermo .raw files can be read on Linux using only the cross-platform
# .NET 8 runtime (via pythonnet's coreclr backend) with the .NET 8 RawFileReader
# DLLs bundled in alpharaw/ext/thermo_fisher/ — no Mono is installed.
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile -t alpharaw-thermo-monofree .
# Run:
#   docker run --rm alpharaw-thermo-monofree
FROM python:3.11-slim-bookworm

# libicu is required for .NET globalization: the reader sets CultureInfo("en-US"),
# which fails under invariant-globalization mode, so ICU must be present.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libicu72 \
    && rm -rf /var/lib/apt/lists/*

# .NET 8 runtime only (deliberately NO Mono).
ENV DOTNET_ROOT=/opt/dotnet
ENV PATH="${DOTNET_ROOT}:${PATH}"
RUN curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && bash /tmp/dotnet-install.sh --channel 8.0 --runtime dotnet --install-dir "${DOTNET_ROOT}" \
    && rm /tmp/dotnet-install.sh \
    && dotnet --list-runtimes

WORKDIR /app
COPY . /app

# pythonnet is pulled in transitively, but pin it explicitly to match the project.
RUN pip install --no-cache-dir pythonnet==3.0.3 \
    && pip install --no-cache-dir .

# coreclr is already the default in clr_utils; set it explicitly to document intent.
ENV ALPHARAW_DOTNET_RUNTIME=coreclr

# Assert Mono is genuinely absent, then run the read test.
CMD ["/bin/sh", "-c", "! command -v mono >/dev/null 2>&1 || (echo 'mono unexpectedly present' && exit 1); python docker/test_thermo_mono_free.py nbs_tests/test_data/iRT.raw"]
