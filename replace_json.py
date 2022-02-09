import sys
import json

enhance = json.load(open('config/enhance.json', 'r'))
ENHANCE = bool(enhance['ENHANCE'])

if not sys.stdin.isatty():
    input_stream = sys.stdin
else:
    try:
        input_filename = sys.argv[1]
    except IndexError:
        message = 'need filename as first argument if stdin is not full'
        raise IndexError(message)
    else:
        input_stream = open(input_filename, 'rU')

json_str = ""
name = ""
started = False
for line in input_stream:
    if line.startswith("INHS") or line.startswith("UWZM"):
        name = line
        print(name)
    if line.startswith("{") and not started:
        started = True
    if started:
        json_str += line
loaded = json.loads(json.dumps(json_str))
print(loaded)
fname = 'metadata.json'
if ENHANCE:
    fname = 'enhanced_' + fname
else:
    fname = 'non_enhanced_' + fname

with open(fname, 'r+') as f:
    metadata = json.load(f)
    if f'/usr/local/bgnn/tulane/{name}' in metadata:# and "errored" in metadata[name] and metadata[name]["errored"]:
        print(metadata[name])
        print(loaded[name])
        metadata[f'/usr/local/bgnn/tulane/{name}'] = loaded[name]
    json.dump(metadata, f)
