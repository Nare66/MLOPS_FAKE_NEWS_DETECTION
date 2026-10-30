import axios from 'axios';

const getBaseUrl = () => {
  return localStorage.getItem('API_BASE_URL') || 'http://localhost:8000';
};

export const getAdminKey = () => {
  return localStorage.getItem('ADMIN_API_KEY') || 'admin-mlops-secret-2026';
};

const createClient = () => {
  return axios.create({
    baseURL: getBaseUrl(),
    timeout: 60000,
    headers: {
      'Content-Type': 'application/json',
    },
  });
};

export const api = {
  healthCheck: async () => {
    const client = createClient();
    const res = await client.get('/');
    return res.data;
  },

  predictNews: async (text) => {
    const client = createClient();
    const res = await client.post('/predict', { text });
    return res.data;
  },

  getHistory: async (limit = 50, offset = 0) => {
    const client = createClient();
    const res = await client.get(`/history?limit=${limit}&offset=${offset}`);
    return res.data;
  },

  getModelInfo: async () => {
    const client = createClient();
    const res = await client.get('/model/info');
    return res.data;
  },

  getMetrics: async () => {
    const client = createClient();
    const res = await client.get('/metrics');
    return res.data;
  },

  uploadDataset: async (file) => {
    const client = createClient();
    const formData = new FormData();
    formData.append('file', file);
    const res = await client.post('/upload-dataset', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  triggerRetraining: async (params = {}) => {
    const client = createClient();
    const adminKey = getAdminKey();
    const res = await client.post('/retrain', params, {
      headers: {
        'X-Admin-Key': adminKey,
      },
    });
    return res.data;
  },

  getRetrainStatus: async () => {
    const client = createClient();
    const res = await client.get('/retrain/status');
    return res.data;
  },

  getMonitoring: async () => {
    const client = createClient();
    const res = await client.get('/monitoring');
    return res.data;
  },

  getMonitoringReportUrl: () => {
    return `${getBaseUrl()}/monitoring/report`;
  },
};
