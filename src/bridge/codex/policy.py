"""Local, user-owned project registry and fail-closed request validation."""
import configparser
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path


class PolicyError(ValueError):
    """Only fixed error codes cross the gateway."""


def text(value, code, maximum=12000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or '\x00' in value:
        raise PolicyError(code)
    return value


def identifier(value):
    text(value, 'invalid_identifier', 128)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise PolicyError('invalid_identifier')
    return value


def private_file(path):
    path = Path(path).expanduser()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PolicyError('private_file_required')
    return path


def token_file(path):
    return text(private_file(path).read_text().strip(), 'invalid_token', 512)


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    path: Path
    readable_roots: tuple[Path, ...] = ()

    def validate(self):
        # Compare against the configured canonical root again at dispatch time.
        if self.path.resolve(strict=True) != self.path or not self.path.is_dir():
            raise PolicyError('project_path_changed')
        # Do not follow directory symlinks while walking. Reject escaping links,
        # even if their target does not yet exist.
        for directory, dirs, files in os.walk(self.path, followlinks=False):
            for name in dirs + files:
                entry = Path(directory) / name
                if entry.is_symlink() and not entry.resolve().is_relative_to(self.path):
                    raise PolicyError('project_symlink_escape')
        return self.path

    def environments(self):
        self.validate()
        ini = self.path / 'platformio.ini'
        parser = configparser.ConfigParser(interpolation=None, strict=False)
        if ini.is_file():
            parser.read(ini)
        return [section[4:] for section in parser.sections() if section.startswith('env:')]


class Registry:
    def __init__(self, filename):
        raw = json.loads(private_file(filename).read_text())
        if not isinstance(raw, dict) or set(raw) != {'projects'} or not isinstance(raw['projects'], list):
            raise PolicyError('invalid_registry')
        self.projects = {}
        for item in raw['projects']:
            if not isinstance(item, dict) or set(item) - {'id', 'name', 'path', 'readable_roots'}:
                raise PolicyError('invalid_registry')
            key = identifier(item['id'])
            path = Path(item['path']).expanduser()
            if not path.is_absolute() or '..' in path.parts or path.resolve(strict=True) != path or path == Path('/'):
                raise PolicyError('invalid_project_path')
            roots = tuple(Path(root).expanduser() for root in item.get('readable_roots', []))
            if any(not root.is_absolute() or root.resolve(strict=True) != root or root == Path('/') for root in roots):
                raise PolicyError('invalid_readable_root')
            if key in self.projects:
                raise PolicyError('duplicate_project')
            self.projects[key] = Project(key, text(item['name'], 'invalid_project_name', 100), path, roots)

    def get(self, key):
        identifier(key)
        if key not in self.projects:
            raise PolicyError('unknown_project')
        project = self.projects[key]
        project.validate()
        return project

    def listing(self):
        items = []
        for project in self.projects.values():
            try:
                envs, available = project.environments(), True
            except (OSError, ValueError):
                envs, available = [], False
            items.append(dict(id=project.id, name=project.name, available=available, environments=envs))
        return {'projects': items}
