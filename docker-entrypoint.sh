#!/bin/sh
# Start as root only long enough to hand /data (named volume, possibly
# created root-owned by older images) to the unprivileged app user, then
# drop privileges for the server itself.
set -e
if [ "$(id -u)" = "0" ]; then
  chown -R app:app /data
  exec setpriv --reuid=app --regid=app --init-groups "$@"
fi
exec "$@"
