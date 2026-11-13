# Copies the keys from your .env into the Azure container app as secrets.
# Run it yourself from the project folder after `az login`:
#     powershell -ExecutionPolicy Bypass -File deploy\set-azure-secrets.ps1
# The values go straight from .env to Azure and are never printed.

$ErrorActionPreference = "Stop"
$app = "support-rag"
$group = "support-rag-rg"

# Run Azure CLI's own Python directly instead of az.cmd: a .cmd file mangles
# characters like & and ? (found in database URLs); this way every value
# reaches Azure exactly as written.
$azPython = Join-Path (Split-Path (Get-Command az).Source) "..\python.exe"
function Invoke-Az { & $azPython -IBm azure.cli @args }

# only double quotes and spaces can't be passed safely as one argument
function Assert-Passable($name, $value) {
    if ($value -match '[\s"]') { throw "$name contains spaces or double quotes; remove them from .env." }
}

# read KEY=VALUE lines from .env (skipping comments and blank lines)
$envFile = Join-Path $PSScriptRoot "..\.env"
$values = @{}
foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$') {
        $values[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
    }
}

# the public site needs Google sign-in, or anyone could use your Gemini key
foreach ($name in "GOOGLE_CLIENT_ID", "SESSION_SECRET", "ALLOWED_EMAILS") {
    if (-not $values[$name]) { throw "Set $name in .env first (see .env.example)." }
}
if ($values["SESSION_SECRET"].Length -lt 32) {
    throw "SESSION_SECRET must be at least 32 characters (see .env.example)."
}
$values["ALLOWED_EMAILS"] = $values["ALLOWED_EMAILS"] -replace '\s', ''
foreach ($name in "GEMINI_API_KEY", "DATABASE_URL") {
    if (-not $values[$name]) { throw "$name is missing from .env." }
}

# environment variable -> secret name in Azure (secret names must be lowercase)
$secretNames = [ordered]@{
    GEMINI_API_KEY      = "gemini-api-key"
    SESSION_SECRET      = "session-secret"
    LANGFUSE_PUBLIC_KEY = "langfuse-public-key"
    LANGFUSE_SECRET_KEY = "langfuse-secret-key"
    DATABASE_URL        = "database-url"
}
$secrets = @()
$envVars = @()
$saved = @()
foreach ($name in $secretNames.Keys) {
    $value = $values[$name]
    if (-not $value) { continue }  # e.g. no Langfuse keys: tracing just stays off
    Assert-Passable $name $value
    $secrets += "$($secretNames[$name])=$value"
    $envVars += "$name=secretref:$($secretNames[$name])"
    $saved += $secretNames[$name]
}

# settings that aren't secret
foreach ($name in "LANGFUSE_BASE_URL", "GENERATION_MODEL", "FALLBACK_MODEL", "MIN_SCORE",
                  "GOOGLE_CLIENT_ID", "ALLOWED_EMAILS", "DAILY_QUESTION_LIMIT") {
    $value = $values[$name]
    if (-not $value) { continue }
    Assert-Passable $name $value
    $envVars += "$name=$value"
}
# the live site: HTTPS-only cookies, and never start without sign-in
$envVars += "COOKIE_SECURE=true"
$envVars += "REQUIRE_LOGIN=true"
# live traces are tagged "production", so local tests don't mix into the metrics page
$envVars += "LANGFUSE_TRACING_ENVIRONMENT=production"

Write-Host "Saving $($secrets.Count) secrets to $app..."
Invoke-Az containerapp secret set --name $app --resource-group $group --secrets @secrets --output none
if ($LASTEXITCODE -ne 0) { throw "Saving the secrets failed." }

Write-Host "Connecting them to the app's environment variables..."
Invoke-Az containerapp update --name $app --resource-group $group --set-env-vars @envVars --output none
if ($LASTEXITCODE -ne 0) { throw "Updating the app failed." }

Write-Host "Done. Secrets set:" ($saved -join ", ")
