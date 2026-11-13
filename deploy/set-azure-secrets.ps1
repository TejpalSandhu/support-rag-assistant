# Copies the keys from your .env into the Azure container app as secrets.
# Run it yourself from the project folder after `az login`:
#     powershell -ExecutionPolicy Bypass -File deploy\set-azure-secrets.ps1
# The values go straight from .env to Azure and are never printed.

$ErrorActionPreference = "Stop"
$app = "support-rag"
$group = "support-rag-rg"

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
if (-not $values["GEMINI_API_KEY"]) {
    throw "GEMINI_API_KEY is missing from .env."
}

# environment variable -> secret name in Azure (secret names must be lowercase)
$secretNames = [ordered]@{
    GEMINI_API_KEY      = "gemini-api-key"
    SESSION_SECRET      = "session-secret"
    LANGFUSE_PUBLIC_KEY = "langfuse-public-key"
    LANGFUSE_SECRET_KEY = "langfuse-secret-key"
}
$secrets = @()
$envVars = @()
$saved = @()
foreach ($name in $secretNames.Keys) {
    $value = $values[$name]
    if (-not $value) { continue }  # e.g. no Langfuse keys: tracing just stays off
    # az is a .cmd script, so characters like & ^ % " would be mangled
    if ($value -notmatch '^[A-Za-z0-9_.:/+=-]+$') {
        throw "$name contains characters that can't be passed safely. Use only letters, digits and - _ . : / + ="
    }
    $secrets += "$($secretNames[$name])=$value"
    $envVars += "$name=secretref:$($secretNames[$name])"
    $saved += $secretNames[$name]
}

# settings that aren't secret
foreach ($name in "LANGFUSE_BASE_URL", "GENERATION_MODEL", "FALLBACK_MODEL", "MIN_SCORE",
                  "GOOGLE_CLIENT_ID", "ALLOWED_EMAILS", "DAILY_QUESTION_LIMIT") {
    $value = $values[$name]
    if (-not $value) { continue }
    if ($value -notmatch '^[A-Za-z0-9_.:/+=@,-]+$') {
        throw "$name contains characters that can't be passed safely."
    }
    $envVars += "$name=$value"
}
# the live site: HTTPS-only cookies, and never start without sign-in
$envVars += "COOKIE_SECURE=true"
$envVars += "REQUIRE_LOGIN=true"
# live traces are tagged "production", so local tests don't mix into the metrics page
$envVars += "LANGFUSE_TRACING_ENVIRONMENT=production"

Write-Host "Saving $($secrets.Count) secrets to $app..."
az containerapp secret set --name $app --resource-group $group --secrets @secrets --output none
if ($LASTEXITCODE -ne 0) { throw "Saving the secrets failed." }

Write-Host "Connecting them to the app's environment variables..."
az containerapp update --name $app --resource-group $group --set-env-vars @envVars --output none
if ($LASTEXITCODE -ne 0) { throw "Updating the app failed." }

Write-Host "Done. Secrets set:" ($saved -join ", ")
