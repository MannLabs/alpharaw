import os

import click

import alpharaw
from alpharaw.ms_data_base import ms_reader_provider
from alpharaw.wrappers import (
    alphapept_wrapper,  # noqa: F401  # TODO remove import side effect => move to register_all_readers()
)

alpharaw.register_all_readers()


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
....................................................
"""
    )
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command(
    "parse",
    help="Convert raw files into alpharaw hdf5 format. A .hdf extension will be added to the input file name(s).",
)
@click.option(
    "--raw_type",
    type=str,
    default="thermo_raw",
    show_default=True,
    help="`thermo_raw` or `sciex_wiff`",
)
@click.option(
    "--raw",
    multiple=True,
    default=[],
    show_default=True,
    help="Raw files, can chained like `--raw raw1 --raw raw2 ...`.",
)
@click.option(
    "--output_dir",
    type=str,
    default="",
    show_default=True,
    help="Folder to write the output .hdf files to. Empty writes next to each raw file.",
)
def _parse(raw_type: str, raw: list, output_dir: str):
    reader = ms_reader_provider.get_reader(raw_type)
    if reader is None:
        print(
            f"{raw_type} is not supported, this may be due to the failed installion of PythonNet or other packages"
        )
    else:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        for raw_file in raw:
            if not os.path.isfile(raw_file):
                print(f"{raw_file} does not exist")
                continue
            reader.import_raw(raw_file)

            hdf_file_path = raw_file + ".hdf"
            if output_dir:
                hdf_file_path = os.path.join(output_dir, hdf_file_path)

            reader.save_hdf(hdf_file_path)
            print(f"Saved {hdf_file_path}")
