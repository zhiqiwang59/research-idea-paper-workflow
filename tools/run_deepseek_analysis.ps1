param(
    [string]$Conferences = "all",
    [int]$Workers = 8,
    [int]$Limit = 0,
    [string]$Model = "deepseek-v4-flash",
    [string]$IdeaFile = ""
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$papersDir = Join-Path $root "papers_2026"
$outDir = Join-Path $root "llm_deepseek_analysis"
$script = Join-Path $PSScriptRoot "deepseek_paper_analysis.py"
if (-not $IdeaFile) {
    $IdeaFile = Join-Path $root "examples\idea.txt"
}

if (-not $env:DEEPSEEK_API_KEY) {
    $secureKey = Read-Host "Enter DEEPSEEK_API_KEY" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

$env:DEEPSEEK_MODEL = $Model
if (-not $env:DEEPSEEK_BASE_URL) {
    $env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
}

python $script `
    --papers-dir $papersDir `
    --idea-file $IdeaFile `
    --out-dir $outDir `
    --conferences $Conferences `
    --workers $Workers `
    --limit $Limit
