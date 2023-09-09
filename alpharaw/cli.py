import click
import os

import alpharaw
from alpharaw.ms_data_base import ms_reader_provider
from alpharaw.legacy_msdata import mgf
from alpharaw.mzml import MzMLReader
from alpharaw.wrappers import alphapept_wrapper
try:
    from alpharaw.sciex import SciexWiffData
    from alpharaw.thermo import ThermoRawData
except (RuntimeError, ImportError):
    print("[WARN] pythonnet is not installed")

@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
)
@click.pass_context
@click.version_option(alpharaw.__version__, "-v", "--version")
def run(ctx, **kwargs):
    click.echo(
r'''
   ___   __     __        ___             
  / _ | / /__  / /  ___ _/ _ \___ __    __
 / __ |/ / _ \/ _ \/ _ `/ , _/ _ `/ |/|/ /
/_/ |_/_/ .__/_//_/\_,_/_/|_|\_,_/|__,__/ 
       /_/                                
....................................................
.{version}.
.{url}.
.{license}.
....................................................
'''.format(
        version=alpharaw.__version__.center(50),
        url=alpharaw.__github__.center(50), 
        license=alpharaw.__license__.center(50),
    )
)
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))

@run.command("parse", help="Convert raw files into alpharaw_hdf format.")
@click.option(
    "--raw_type", type=str, default="thermo_raw",
    show_default=True, help=f"Only `thermo_raw` is supported currently.",
)
@click.option(
    "--raw", multiple=True, default=[],
    show_default=True, help="Raw files, can be `--raw raw1 --raw raw2 ...`."
)
def _parse(raw_type:str, raw:list):
    reader = ms_reader_provider.get_reader(raw_type)
    if reader is None: 
        print(f"{raw_type} is not supported, this may be due to the failed installion of PythonNet or other packages")
    else:
        for raw_file in raw:
            if not os.path.isfile(raw_file):
                print(f"{raw_file} does not exist")
                continue
            reader.import_raw(raw_file)
