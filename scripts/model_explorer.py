"""Local, offline saved-trace explorer. No model import, GPU, or inference path."""
import argparse
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_explorer import Catalog, Explorer, csvrows, readjson
from src.evaluation.common import private_directory

UI = Path(__file__).with_name('model_explorer')


def make_handler(catalog):
    explorer = Explorer(catalog)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(route.query).items()}
            def integer(name, default):
                return int(q.get(name, default))
            try:
                if route.path == '/api/catalog':
                    result = explorer.catalog_data()
                elif route.path == '/api/context':
                    result = explorer.describe(q['id'])
                elif route.path == '/api/comparison':
                    result = explorer.comparison(q['a'], q['b'], q.get('site', 'resid'))
                elif route.path == '/api/vectors':
                    result = explorer.vectors(q['a'], q['b'], integer('layer', 0), integer('j', 0),
                        q.get('site', 'resid'), integer('start', 0), integer('end', 3072))
                elif route.path == '/api/embeddings':
                    result = explorer.embeddings(q['a'], q['b'], integer('ia', 0), integer('ib', 0))
                elif route.path == '/api/predictions':
                    result = explorer.predictions(q['a'], q['b'], integer('j', 0))
                elif route.path == '/api/geometry':
                    result = explorer.geometry(q['a'], q['b'], [int(v) for v in q['layers'].split(',')], q.get('site', 'resid'))
                elif route.path == '/api/decoder-info':
                    result = explorer.decoder_info(q['history'])
                elif route.path == '/api/decoder':
                    result = explorer.decoder(q['history'], q.get('site', 'z_q'), integer('frame', 0),
                        integer('subframe', 0), integer('channel', 0), integer('chstart', 0), integer('chcount', 32),
                        int(q['tstart']) if 'tstart' in q else None, integer('tcount', 160))
                elif route.path == '/api/waveform':
                    result = explorer.waveform(q['history'])
                elif route.path == '/audio':
                    record = explorer.one('recording', q['id'])
                    return self.file(Path(record['path']), 'audio/wav')
                elif route.path in ('/', '/app.js', '/style.css'):
                    name = 'index.html' if route.path == '/' else route.path[1:]
                    return self.file(UI / name, mimetypes.guess_type(name)[0] or 'text/plain')
                else:
                    self.send_error(404)
                    return
                body = json.dumps(result, allow_nan=False, separators=(',', ':')).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError, OSError, IndexError) as exc:
                body = json.dumps({'error': str(exc)}).encode()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def file(self, path, mime):
            data = path.read_bytes()
            start, end = 0, len(data)-1
            requested = self.headers.get('Range')
            if requested and requested.startswith('bytes='):
                text = requested[6:].split(',')[0]
                lo, hi = text.split('-')
                start, end = int(lo or 0), int(hi) if hi else end
                if not 0 <= start <= end < len(data):
                    self.send_error(416)
                    return
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type', mime + ('; charset=utf-8' if mime.startswith('text/') or 'javascript' in mime else ''))
            self.send_header('Content-Length', str(end-start+1))
            self.send_header('Accept-Ranges', 'bytes')
            if requested:
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
            self.end_headers()
            self.wfile.write(data[start:end+1])

        def log_message(self, *args):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('register')
    r.add_argument('--catalog', type=Path, required=True)
    r.add_argument('--trace', type=Path, action='append', required=True)
    r.add_argument('--experiment', required=True)
    r.add_argument('--dataset', required=True)
    r.add_argument('--language', required=True)
    r.add_argument('--settings', type=Path)
    r.add_argument('--acoustics', type=Path)
    r = sub.add_parser('register-condition')
    r.add_argument('--catalog', type=Path, required=True)
    r.add_argument('--spec', type=Path, required=True,
        help='JSON: experiment_id, label, metadata. Metadata-only registration, not an executed trace.')
    r = sub.add_parser('register-manifest')
    r.add_argument('--catalog', type=Path, required=True)
    r.add_argument('--manifest', type=Path, required=True, help='Versioned, hash-declared condition-agnostic capture manifest')
    r = sub.add_parser('verify')
    r.add_argument('--catalog', type=Path, required=True)
    r = sub.add_parser('serve')
    r.add_argument('--catalog', type=Path, required=True)
    r.add_argument('--port', type=int, default=8766)
    r.add_argument('--open', action='store_true', help='Open the local browser after source verification')
    args = p.parse_args()
    private_directory(args.catalog.parent)
    if args.action not in ('register','register-manifest') and not args.catalog.is_file():
        p.error('Register real traces first; catalog does not exist')
    catalog = Catalog(args.catalog)
    try:
        if args.action == 'register':
            settings = readjson(args.settings)['decoding'] if args.settings else None
            acoustic = csvrows(args.acoustics) if args.acoustics else []
            for path in args.trace:
                print(json.dumps(catalog.register_legacy(path, args.experiment, args.dataset, args.language, settings, acoustic)))
        elif args.action == 'register-condition':
            spec = readjson(args.spec)
            cid = catalog.register_condition(spec['experiment_id'], spec['label'], spec['metadata'])
            catalog.db.commit()
            print(json.dumps({'condition_id': cid, 'trace_status': 'metadata only; no results'}))
        elif args.action == 'register-manifest':
            print(json.dumps(catalog.register_spec(args.manifest)))
        else:
            verified = catalog.verify_sources()
            print(json.dumps({'verified_sources': len(verified), 'verified_artifacts': sum(r['verified_artifacts'] for r in verified)}), flush=True)
    finally:
        catalog.close()
    if args.action == 'serve':
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(args.catalog))
        print(f'Explorer: http://127.0.0.1:{args.port} (offline, read-only trace APIs; Ctrl+C stops)', flush=True)
        if args.open:
            import threading
            import webbrowser
            threading.Timer(.3,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
