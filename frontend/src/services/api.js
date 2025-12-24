import axios from 'axios'

// 在生產環境中，使用相對路徑通過 nginx 代理
// 在開發環境中，使用完整 URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
  (import.meta.env.PROD ? '/api/v1' : 'http://localhost:8000/api/v1')

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 圖像相關 API
export const imageAPI = {
  upload: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post('/images/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },
  
  get: async (imageId) => {
    const response = await api.get(`/images/${imageId}`)
    return response.data
  },
  
  getFile: (imageId) => {
    return `${API_BASE_URL}/images/${imageId}/file`
  },
  
  delete: async (imageId) => {
    await api.delete(`/images/${imageId}`)
  },
  
  detectSeal: async (imageId) => {
    const response = await api.post(`/images/${imageId}/detect-seal`)
    return response.data
  },
  
  updateSealLocation: async (imageId, locationData) => {
    const response = await api.put(`/images/${imageId}/seal-location`, locationData)
    return response.data
  },
  
  // 多印鑑檢測相關 API（測試功能）
  detectMultipleSeals: async (imageId, maxSeals = 10) => {
    const response = await api.post(`/images/${imageId}/detect-multiple-seals`, null, {
      params: { max_seals: maxSeals }
    })
    return response.data
  },
  
  saveMultipleSeals: async (imageId, seals) => {
    const response = await api.post(`/images/${imageId}/save-multiple-seals`, {
      seals: seals.map(seal => ({
        bbox: seal.bbox,
        center: seal.center,
        confidence: seal.confidence || 0.5
      }))
    })
    return response.data
  },
  
  cropSeals: async (imageId, seals, margin = 10) => {
    const response = await api.post(`/images/${imageId}/crop-seals`, {
      seals: seals.map(seal => ({
        bbox: seal.bbox,
        center: seal.center,
        confidence: seal.confidence || 0.5
      })),
      margin: margin
    })
    return response.data
  },
  
  // 多印鑑比對相關 API（測試功能）
  // timeout: 超時時間（毫秒），根據印鑑數量動態計算
  compareImage1WithSeals: async (image1Id, sealImageIds, threshold = 0.95, similaritySsimWeight = 0.5, similarityTemplateWeight = 0.35, pixelSimilarityWeight = 0.1, histogramSimilarityWeight = 0.05, timeout = null) => {
    // 如果沒有指定超時時間，根據印鑑數量動態計算
    // 基礎超時時間：30秒，每個印鑑增加15秒
    const calculatedTimeout = timeout || (30000 + (sealImageIds?.length || 0) * 15000)
    
    const response = await api.post(`/images/${image1Id}/compare-with-seals`, {
      seal_image_ids: sealImageIds,
      threshold: threshold,
      similarity_ssim_weight: similaritySsimWeight,
      similarity_template_weight: similarityTemplateWeight,
      pixel_similarity_weight: pixelSimilarityWeight,
      histogram_similarity_weight: histogramSimilarityWeight
    }, {
      timeout: calculatedTimeout
    })
    return response.data
  },
}

// 比對相關 API
export const comparisonAPI = {
  create: async (data) => {
    const response = await api.post('/comparisons/', data)
    return response.data
  },
  
  get: async (comparisonId) => {
    const response = await api.get(`/comparisons/${comparisonId}`)
    return response.data
  },
  
  getStatus: async (comparisonId) => {
    const response = await api.get(`/comparisons/${comparisonId}/status`)
    return response.data
  },
  
  list: async (skip = 0, limit = 100, includeDeleted = false) => {
    const response = await api.get('/comparisons/', {
      params: { skip, limit, include_deleted: includeDeleted },
    })
    return response.data
  },
  
  update: async (comparisonId, data) => {
    const response = await api.put(`/comparisons/${comparisonId}`, data)
    return response.data
  },
  
  delete: async (comparisonId) => {
    await api.delete(`/comparisons/${comparisonId}`)
  },
  
  restore: async (comparisonId) => {
    const response = await api.post(`/comparisons/${comparisonId}/restore`)
    return response.data
  },
  
  retry: async (comparisonId, enableRotationSearch = true, enableTranslationSearch = true) => {
    const response = await api.post(`/comparisons/${comparisonId}/retry?enable_rotation_search=${enableRotationSearch}&enable_translation_search=${enableTranslationSearch}`)
    return response.data
  },
}

// 視覺化相關 API
export const visualizationAPI = {
  getComparisonImage: (comparisonId) => {
    return `${API_BASE_URL}/comparisons/${comparisonId}/comparison-image`
  },
  
  getHeatmap: (comparisonId) => {
    return `${API_BASE_URL}/comparisons/${comparisonId}/heatmap`
  },
  
  getOverlay: (comparisonId, overlayType = '1') => {
    return `${API_BASE_URL}/comparisons/${comparisonId}/overlay?overlay_type=${overlayType}`
  },
}

// 統計相關 API
export const statisticsAPI = {
  get: async () => {
    const response = await api.get('/statistics/')
    return response.data
  },
}

export default api

