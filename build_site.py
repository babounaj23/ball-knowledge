"""turns site_data.json + site_template.html into index.html (the actual website)"""
import json

data = json.load(open("site_data.json"))
page = open("site_template.html").read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Match predictions for 8 European leagues with the full math behind every pick.">
<style>[hidden]{{display:none!important}} img{{max-width:100%}}</style>
</head>
<body>
{page}
</body>
</html>
"""
open("index.html", "w").write(html)
print("wrote index.html", len(html), "bytes")
