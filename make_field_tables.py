"""
Usage:

  >>> run make_field_tables
  >>> lines = update_docstrings()
  >>> open('new_models.py', 'w').writelines(lines)
  % meld new_models.py kadi/events/models.py
  % mv new_models.py kadi/events/models.py
"""
import re
import textwrap

from astropy.table import Table
from kadi.events.models import get_event_models, BaseModel
import kadi.events.models

models = get_event_models(baseclass=BaseModel)


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
    out = Table(zip(*rows), names=('Field', 'Type', 'Description'))
    out_lines = out.pformat()
    sep = out_lines[1].replace('-', '=')
    out_lines[1] = sep
    out_lines.append(sep)
    out_lines.insert(0, sep)
    return out_lines


def get_docstring(model, docstring=None):
    lines = []
    if docstring is None:
        docstring = textwrap.dedent(model.__doc__).strip()
    doc_lines = docstring.splitlines()
    idxs = [ii for ii, line in enumerate(doc_lines) if re.match(r'^[ =]+$', line)]
    print 'idxs', idxs
    if len(idxs) > 2:
        for idx, idx2 in zip(idxs, idxs[2:]):
            print idx
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


def update_docstrings():
    model_file = re.sub(r'\.pyc$', '.py', kadi.events.models.__file__)
    print model_file
    lines = open(model_file, 'r').readlines()
    for model in models.values():
        class_start = 'class {}('.format(model.__name__)
        idxs = [ii for ii, line in enumerate(lines) if line.strip() == '"""']
        print class_start
        for idx, idx2 in zip(idxs, idxs[1:]):
            if lines[idx - 1].startswith(class_start):
                print 'dropping lines', idx + 1, idx2
                docstring_lines = ['    {}\n'.format(x.rstrip()) for x in get_docstring(model)]
                docstring_lines = [x if x.strip() else '\n' for x in docstring_lines]
                lines = lines[:idx + 1] + docstring_lines + lines[idx2:]
                break
    return lines

if __name__ == '__main__':
    print 'duh'
