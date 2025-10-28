param(
    [string]$Device = ""
)

python -m src.app --once $(if($Device -ne ""){"--device `"$Device`""})
