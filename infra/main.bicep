@description('Location for all resources')
param location string = resourceGroup().location

@description('Prefix for resource names')
param prefix string = 'warwickprep'

@description('Azure OpenAI model deployments')
param chatModelName string = 'gpt-4o-mini'
param embeddingModelName string = 'text-embedding-3-small'
@description('Whether to create Azure OpenAI model deployments from this template')
param deployOpenAiModelDeployments bool = false

@description('App Service plan SKU name')
param appServicePlanSkuName string = 'B1'

@description('App Service plan SKU tier')
param appServicePlanSkuTier string = 'Basic'
@description('Whether to deploy App Service hosting resources')
param deployWebApp bool = true

var uniqueSuffix = toLower(uniqueString(resourceGroup().id))
var storageAccountName = toLower('wsc${take(replace('${prefix}${uniqueSuffix}', '-', ''), 21)}')
var searchServiceName = '${prefix}-${uniqueSuffix}-search'
var openAiAccountName = '${prefix}-${uniqueSuffix}-openai'
var docIntelligenceName = '${prefix}-${uniqueSuffix}-docintel'
var appServicePlanName = '${prefix}-${uniqueSuffix}-plan'
var webAppName = '${prefix}-${uniqueSuffix}-web'

// ── Storage Account ──────────────────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource rawContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'webcrawl-raw'
  properties: { publicAccess: 'None' }
}

resource pdfContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'webcrawl-pdf'
  properties: { publicAccess: 'None' }
}

// ── Azure AI Search ───────────────────────────────────────────
resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchServiceName
  location: location
  sku: { name: 'basic' }
  properties: {
    replicaCount: 1
    partitionCount: 1
    publicNetworkAccess: 'enabled'
  }
}

// ── Azure OpenAI ──────────────────────────────────────────────
resource openAiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAiAccountName
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    publicNetworkAccess: 'enabled'
    customSubDomainName: openAiAccountName
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (deployOpenAiModelDeployments) {
  parent: openAiAccount
  name: chatModelName
  sku: { name: 'Standard', capacity: 20 }
  properties: {
    model: {
      format: 'OpenAI'
      name: chatModelName
      version: '2024-07-18'
    }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = if (deployOpenAiModelDeployments) {
  parent: openAiAccount
  name: embeddingModelName
  sku: { name: 'Standard', capacity: 20 }
  properties: {
    model: {
      format: 'OpenAI'
      name: embeddingModelName
      version: '1'
    }
  }
  dependsOn: [chatDeployment]
}

// ── Azure AI Document Intelligence ───────────────────────────
resource docIntelligence 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: docIntelligenceName
  location: location
  kind: 'FormRecognizer'
  sku: { name: 'S0' }
  properties: {
    publicNetworkAccess: 'enabled'
    customSubDomainName: docIntelligenceName
  }
}

// ── Azure App Service ─────────────────────────────────────────
resource appServicePlan 'Microsoft.Web/serverfarms@2023-12-01' = if (deployWebApp) {
  name: appServicePlanName
  location: location
  sku: {
    name: appServicePlanSkuName
    tier: appServicePlanSkuTier
  }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

resource webApp 'Microsoft.Web/sites@2023-12-01' = if (deployWebApp) {
  name: webAppName
  location: location
  kind: 'app,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    clientAffinityEnabled: false
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      alwaysOn: true
      appCommandLine: 'gunicorn -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000 src.api.main:app'
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
    }
  }
}

var openAiApiKey = openAiAccount.listKeys().key1
var docIntelApiKey = docIntelligence.listKeys().key1
var searchApiKey = searchService.listAdminKeys().primaryKey
var storageKey = storageAccount.listKeys().keys[0].value
var storageConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageKey};EndpointSuffix=${environment().suffixes.storage}'

resource webAppAppSettings 'Microsoft.Web/sites/config@2023-12-01' = if (deployWebApp) {
  parent: webApp
  name: 'appsettings'
  properties: {
    SCM_DO_BUILD_DURING_DEPLOYMENT: 'true'
    ENABLE_ORYX_BUILD: 'true'
    PYTHONPATH: '/home/site/wwwroot'
    AZURE_OPENAI_ENDPOINT: openAiAccount.properties.endpoint
    AZURE_OPENAI_API_KEY: openAiApiKey
    AZURE_OPENAI_CHAT_DEPLOYMENT: chatModelName
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: embeddingModelName
    AZURE_SEARCH_ENDPOINT: 'https://${searchService.name}.search.windows.net'
    AZURE_SEARCH_API_KEY: searchApiKey
    AZURE_SEARCH_INDEX_NAME: 'warwickprep-content'
    AZURE_DOC_INTELLIGENCE_ENDPOINT: docIntelligence.properties.endpoint
    AZURE_DOC_INTELLIGENCE_KEY: docIntelApiKey
    AZURE_STORAGE_ACCOUNT_NAME: storageAccount.name
    AZURE_STORAGE_CONNECTION_STRING: storageConnectionString
    AZURE_STORAGE_CONTAINER_RAW: rawContainer.name
    AZURE_STORAGE_CONTAINER_PDF: pdfContainer.name
    CRAWL_START_URL: 'https://www.warwickprep.com/'
    CRAWL_ALLOWED_DOMAINS: 'warwickprep.com'
    CRAWL_MAX_DEPTH: '5'
    CRAWL_DELAY_SECONDS: '1'
    API_HOST: '0.0.0.0'
    API_PORT: '8000'
    RAG_TOP_K: '5'
    RAG_CHUNK_SIZE: '512'
    RAG_CHUNK_OVERLAP: '64'
  }
}

// ── Outputs ───────────────────────────────────────────────────
output storageAccountName string = storageAccount.name
output searchEndpoint string = 'https://${searchService.name}.search.windows.net'
output openAiEndpoint string = openAiAccount.properties.endpoint
output docIntelEndpoint string = docIntelligence.properties.endpoint
output webAppName string = deployWebApp ? webApp.name : ''
output webAppUrl string = deployWebApp ? 'https://${webApp.properties.defaultHostName}' : ''
