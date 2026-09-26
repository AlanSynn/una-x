#!/usr/bin/env python3
import json, subprocess
def run(*a):
    p=subprocess.run(a,text=True,capture_output=True)
    return {"returncode":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip()}
out={
"head":run("git","rev-parse","HEAD"),
"branch":run("git","branch","--show-current"),
"status":run("git","status","--porcelain=v1"),
"remotes":run("git","remote","-v"),
"main":run("git","rev-parse","main"),
}
print(json.dumps(out,indent=2))
