# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Auto-generate event class documenation based on sphinx model definitions.

- update_models_docstrings
- make_events_table
- make_event_descriptions_section

Usage:

ipython
run make_field_tables

>>> update_model_docstrings(outfile='models_test.py')
% diff models_test.py ../kadi/events/models.py
% mv models_test.py ../kadi/events/models.py
# Or just set outfile to ../kadi/events/models.py straight away

>>> make_event_descriptions_section('event_descriptions.rst')

"""
import sys
import os
import re
import textwrap

from astropy.table import Table

sys.path.insert(0, os.path.abspath('..'))
from kadi.events.models import get_event_models, BaseModel
import kadi.events.models

models = get_event_models(baseclass=BaseModel)


def table_to_rst(dat):
    out_lines = dat.pformat(max_lines=-1, max_width=-1)
    sep = out_lines[1].replace('-', '=')
    out_lines[1] = sep
    out_lines.append(sep)
    out_lines.insert(0, sep)
    return [x.rstrip() for x in out_lines]


def get_fields_table(model):
    rows = []
    for field in model._meta.fields:
        field_type = field.__class__.__name__
        if field_type == 'AutoField':
            continue
        if field_type.endswith('Field'):
            field_type = field_type[:-5]
        if field_type == 'Char':
            field_type += '({})'.format(field.max_length)
        row = [' {} '.format(x) for x in (field.name, field_type, field.help_text)]
        rows.append(row)
    dat = Table(zip(*rows), names=('Field', 'Type', 'Description'))
    out_lines = table_to_rst(dat)
    return out_lines


def write_lines(lines, outfile):
    out = '\n'.join(lines) + '\n'
    if outfile:
        with open(outfile, 'w') as f:
            f.write(out)
    else:
        print out


def get_docstring(model, docstring=None):
    """
    Get the docstring from ``model`` and update with the appropriate
    model fields.  This will either insert a new table or repace an
    existing table.
    """
    lines = []
    if docstring is None:
        docstring = textwrap.dedent(model.__doc__).strip()
    doc_lines = docstring.splitlines()
    idxs = [ii for ii, line in enumerate(doc_lines) if re.match(r'^[ =]+$', line)]
    if len(idxs) > 2:
        for idx, idx2 in zip(idxs, idxs[2:]):
            # Chop existing fields table
            if doc_lines[idx - 2] == '**Fields**':
                doc_lines = doc_lines[:idx - 1] + doc_lines[idx2 + 1:]
                break

    for doc_line in doc_lines:
        lines.append(doc_line)
        if doc_line == '**Fields**':
            lines.append('')
            lines.extend(get_fields_table(model))

    return lines


def update_models_docstrings(outfile=None):
    """
    Update the doc strings in kadi/events/models.py based on current
    model field definitions.
    """
    model_file = re.sub(r'\.pyc$', '.py', kadi.events.models.__file__)
    lines = open(model_file, 'r').read().splitlines()
    for model in models.values():
        class_start = 'class {}('.format(model.__name__)
        idxs = [ii for ii, line in enumerate(lines) if line.strip() == '"""']
        for idx, idx2 in zip(idxs, idxs[1:]):
            if lines[idx - 1].startswith(class_start):
                docstring_lines = ['    {}'.format(x.rstrip()) for x in get_docstring(model)]
                docstring_lines = [x if x.strip() else '' for x in docstring_lines]
                lines = lines[:idx + 1] + docstring_lines + lines[idx2:]
                break
    write_lines(lines, outfile)


def make_event_descriptions_section(outfile=None):
    """
    Make the Event Descriptions section of the Kadi RST docs.
    """
    out_lines = ['=======================',
                 'Event Descriptions',
                 '=======================',
                 '']

    for model_name in sorted(models):
        model = models[model_name]
        docstring = textwrap.dedent(model.__doc__).strip()
        lines = docstring.splitlines()
        out_lines.append('.. _event_{}:\n'.format(model_name))
        out_lines.append(lines[0])
        out_lines.append('-' * len(lines[0]) + '\n')
        out_lines.extend(lines[2:])
        out_lines.append('')
    write_lines(out_lines, outfile)


def make_events_tables(outfile=None):
    """
    Make the summary table of event classes and descriptions.
    """
    rows = []
    out_lines = []
    for model_name in sorted(models):
        model = models[model_name]
        query_name = model_name + 's'  # See query.py main
        out_lines.append('.. |{0}| replace:: :class:`~kadi.events.models.{0}`'
                         .format(model.__name__))
        row = ('|{}|'.format(model.__name__),
               ':ref:`event_{}`'.format(model_name),
               '``{}``'.format(query_name))
        rows.append(row)
    dat = Table(zip(*rows), names=('Event class', 'Description', 'Query name'))
    out_lines.append('')
    out_lines.extend(table_to_rst(dat))

    rows = []
    for model_name in sorted(models):
        model = models[model_name]
        docstring = textwrap.dedent(model.__doc__).strip()
        lines = docstring.splitlines()
        description = lines[0].strip()
        query_name = model_name + 's'  # See query.py main
        row = (query_name, description, model.__name__)
        rows.append(row)

    dat = Table(zip(*rows), names=['Query name', 'Description', 'Event class'])
    out_lines.append('')
    out_lines.extend(table_to_rst(dat))

    write_lines(out_lines, outfile)
