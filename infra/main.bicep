@description('Location for all resources')
param location string = resourceGroup().location

@description('Prefix for resource names')
param prefix string = 'warwickprep'

@description('Azure OpenAI model deployments')
param chatModelName string = 'gpt-4o-mini'
param embeddingModelName string = 'text-embedding-3-small'

// ── Storage Account ──────────────────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${prefix}storage${uniqueString(resourceGroup().id)}'
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
  name: '${prefix}-search'
  location: location
  sku: { name: 'basic' }
  properties: {
    replicaCount: 1
    partitionCount: 1
    publicNetworkAccess: 'Enabled'
  }
}

// ── Azure OpenAI ──────────────────────────────────────────────
resource openAiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${prefix}-openai'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: '${prefix}-openai'
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
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

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
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
  name: '${prefix}-docintel'
  location: location
  kind: 'FormRecognizer'
  sku: { name: 'S0' }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: '${prefix}-docintel'
  }
}

// ── Outputs ───────────────────────────────────────────────────
output storageAccountName string = storageAccount.name
output searchEndpoint string = 'https://${searchService.name}.search.windows.net'
output openAiEndpoint string = openAiAccount.properties.endpoint
output docIntelEndpoint string = docIntelligence.properties.endpoint
