import yaml
import argparse


class YamlStructure(dict):
    def __init__(self, data):
        super().__init__()
        assert isinstance(data, dict), "Check Data Type"
        self.update(data)

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]

    def __repr__(self):
        def _recur(string, d, cnt):
            for k, v in d.items():
                if cnt == 0:
                    string.append('\n')
                if isinstance(v, dict):
                    string.append('  ' * cnt +  f'{k} :')
                    _recur(string, v, cnt + 1)
                elif isinstance(v, list):
                    string.append('  ' * cnt +  f'{k} :')
                    for item in v:
                        string.append('  ' * (cnt + 1) +  f'- {item}')
                else:
                    string.append('  ' * cnt + f'{k} : {v}')
            return '\n'.join(string)

        string = []
        return _recur(string, self, 0)


class Parser:
    def __init__(self, path, args=None):
        if args is not None:
            raise NotImplementedError("Don't use args")
            assert isinstance(args, argparse.Namespace), "Check args"

        self.init_yaml(path)

    def init_yaml(self, path):
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.C = YamlStructure(data)

        def to_structure(d):
            for k, v in d.items():
                if isinstance(v, (YamlStructure, dict)):
                    d[k] = YamlStructure(v)
                    to_structure(v)

        to_structure(self.C)

    def dump(self, path=None):
        tmp = {}

        def update(src, dst):
            for k, v in src.items():
                if isinstance(v, dict):
                    dst[k] = {}
                    update(v, dst[k])
                else:
                    dst[k] = v

        update(self.C, tmp)

        if path is None:
            yaml.dump(tmp)
            print()
        else:
            with open(path, 'w') as f:
                yaml.dump(tmp, f)
        return tmp

    def update_mem(self, *key, value):
        d = self.C
        for k in key:
            if k not in d.keys():
                d[k] = value
            elif isinstance(d[k], dict):
                d = d[k]
            else:
                d[k] = value
