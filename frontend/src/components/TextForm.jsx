import React, { useState } from 'react';
import { ClipLoader } from 'react-spinners';
import { motion } from 'framer-motion';

const TextForm = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const isURL = (str) => {
    const pattern = /^(https?:\/\/|www\.)|(\.(com|net|org|co|xyz|info|biz|online|ru|io|shop)\b)/i;
    return pattern.test(str.trim());
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Sunucuya ulaşılamadı.' });
    }

    setLoading(false);
  };

  const formatModelResult = (value) => {
    if (value === 1) return 'SPAM';
    if (value === 0) return 'Spam Değil';
    return '—';
  };

  return (
    <div className="text-form-container">
      <h1 style={{ textAlign: 'center' }}>📨 Phishing Detection Tool</h1>
      <form onSubmit={handleSubmit} style={{ marginBottom: 30 }}>
        <textarea
          rows="6"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Metni veya URL'yi buraya yapıştırın..."
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Kontrol ediliyor...' : 'Gönder'}
        </button>
      </form>

      {loading && (
        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <ClipLoader color="#0077cc" size={40} />
        </div>
      )}

      {result && (
        <motion.div
          className="result-box"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {result.error ? (
            <p style={{ color: 'red' }}>{result.error}</p>
          ) : (
            <>
              <p><strong>Sonuç:</strong> {result.final_result === 'spam' ? '🚨 SPAM' : '✅ Spam Değil'}</p>
              <p><strong>Yöntem:</strong> {result.method}</p>

              {result.models && (
                <ul style={{ lineHeight: '1.8' }}>
                  <li>🔍 Rule-Based: {formatModelResult(result.models.rule_based)}</li>
                  {!isURL(text) && (
                    <>
                      <li>🇹🇷 Türkçe Naive Bayes: {formatModelResult(result.models.turkish_nb)}</li>
                      <li>🇬🇧 Combined ML (LightGBM): {formatModelResult(result.models.combined_ml)}</li>
                      <li>🤖 Combined DL (LSTM): {formatModelResult(result.models.combined_dl)}</li>
                      <li>📧 CEAS ML: {formatModelResult(result.models.ceas_ml)}</li>
                      <li>📧 CEAS DL: {formatModelResult(result.models.ceas_dl)}</li>
                    </>
                  )}
                  {isURL(text) && (
                    <>
                      <li>🔗 URL ML: {formatModelResult(result.models.url_ml)}</li>
                      <li>🔗 URL DL: {formatModelResult(result.models.url_dl)}</li>
                    </>
                  )}
                </ul>
              )}
            </>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default TextForm;
