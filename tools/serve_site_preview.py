"""Serve docs locally under the same project prefix as GitHub Pages."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--prefix", default="/gnss_gpu/")
    args = parser.parse_args()
    prefix = "/" + args.prefix.strip("/") + "/"
    if prefix == "//" or any(
        part in ("..", ".claude") for part in prefix.casefold().split("/")
    ):
        parser.error("A nonempty project prefix is required")
    docs = Path(__file__).resolve().parents[1] / "docs"

    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript"}

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if ".claude" in path.replace("\\", "/").casefold().split("/"):
                self.send_error(404)
                return
            if path == "/":
                self.send_response(302)
                self.send_header("Location", prefix)
                self.end_headers()
                return
            if not path.startswith(prefix):
                self.send_error(404, "Use the project prefix")
                return
            self.path = self.path[len(prefix) - 1 :]
            super().do_GET()

        def do_HEAD(self):
            # Reuse GET routing, suppressing the body through the standard method.
            path = unquote(urlsplit(self.path).path)
            if path == "/":
                self.send_response(302)
                self.send_header("Location", prefix)
                self.end_headers()
                return
            if not path.startswith(prefix) or ".claude" in path.replace(
                "\\", "/"
            ).casefold().split("/"):
                self.send_error(404)
                return
            self.path = self.path[len(prefix) - 1 :]
            super().do_HEAD()

    class PreviewServer(ThreadingHTTPServer):
        # Module imports can open a burst of connections during browser tests.
        request_queue_size = 128

    handler = partial(Handler, directory=str(docs))
    with PreviewServer(("127.0.0.1", args.port), handler) as server:
        print(
            f"Preview: http://127.0.0.1:{args.port}{prefix}simulation.html", flush=True
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
