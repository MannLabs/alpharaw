import os

import click

import alpharaw
from alpharaw.legacy_msdata import mgf  # noqa: F401  # TODO remove import side effect
from alpharaw.ms_data_base import ms_reader_provider
from alpharaw.mzml import MzMLReader  # noqa: F401  # TODO remove import side effect
from alpharaw.wrappers import (
    alphapept_wrapper,  # noqa: F401  # TODO remove import side effect
)

try:
    from alpharaw.sciex import (
        SciexWiffData,  # noqa: F401 # TODO remove import side effect
    )
    from alpharaw.thermo import (
        ThermoRawData,  # noqa: F401 # TODO remove import side effect
    )
except (RuntimeError, ImportError):
    print("[WARN] pythonnet is not installed")


@click.group(
    context_settings=dict(
        help_option_names=["-h", "--help"],
    ),
    invoke_without_command=True,
)
@click.pass_context
@click.version_option(alpharaw.__version__, "-v", "--version")
def run(ctx, **kwargs):
    click.echo(
        rf"""
   ___   __     __        ___
  / _ | / /__  / /  ___ _/ _ \___ __    __
 / __ |/ / _ \/ _ \/ _ `/ , _/ _ `/ |/|/ /
/_/ |_/_/ .__/_//_/\_,_/_/|_|\_,_/|__,__/
       /_/
....................................................
.{alpharaw.__version__.center(50)}.
.{alpharaw.__github__.center(50)}.
.{alpharaw.__license__.center(50)}.
....................................................
"""
    )
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("parse", help="Convert raw files into alpharaw hdf5 (.hdf) format.")
@click.option(
    "--raw_type",
    type=str,
    default="thermo_raw",
    show_default=True,
    help="Only `thermo_raw`, `sciex_wiff` is supported currently.",
)
@click.option(
    "--raw",
    multiple=True,
    default=[],
    show_default=True,
    help="Raw files, can be `--raw raw1 --raw raw2 ...`.",
)
def _parse(raw_type: str, raw: list):
    reader = ms_reader_provider.get_reader(raw_type)
    if reader is None:
        print(
            f"{raw_type} is not supported, this may be due to the failed installion of PythonNet or other packages"
        )
    else:
        for raw_file in raw:
            if not os.path.isfile(raw_file):
                print(f"{raw_file} does not exist")
                continue
            reader.import_raw(raw_file)
