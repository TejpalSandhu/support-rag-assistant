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

# the public site must be password protected, or anyone could use your Gemini key
if (-not $values["APP_ACCESS_KEY"]) {
    throw "Set APP_ACCESS_KEY in .env first (this is the password for the website)."
}
if (-not $values["GEMINI_API_KEY"]) {
    throw "GEMINI_API_KEY is missing from .env."
}

# environment variable -> secret name in Azure (secret names must be lowercase)
$secretNames = [ordered]@{
    GEMINI_API_KEY      = "gemini-api-key"
    APP_ACCESS_KEY      = "app-access-key"
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
foreach ($name in "LANGFUSE_BASE_URL", "GENERATION_MODEL") {
    if ($values[$name]) { $envVars += "$name=$($values[$name])" }
}

Write-Host "Saving $($secrets.Count) secrets to $app..."
az containerapp secret set --name $app --resource-group $group --secrets @secrets --output none
if ($LASTEXITCODE -ne 0) { throw "Saving the secrets failed." }

Write-Host "Connecting them to the app's environment variables..."
az containerapp update --name $app --resource-group $group --set-env-vars @envVars --output none
if ($LASTEXITCODE -ne 0) { throw "Updating the app failed." }

Write-Host "Done. Secrets set:" ($saved -join ", ")
