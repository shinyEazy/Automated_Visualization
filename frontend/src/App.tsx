import React, { useState } from 'react';
import axios from 'axios';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';

interface ApiResult {
  [key: string]: any;
}

const App: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<ApiResult | null>(null);

  const [inputText, setInputText] = useState<string>('');
  
  const TextInputSection = () => (
    <Card className="p-6">
      <div className="space-y-4">
        <div>
          <Label htmlFor="text-input" className="text-sm font-medium text-gray-700">
            Enter Text for Analysis
          </Label>
          <Textarea
            id="text-input"
            placeholder="Type your text here..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            className="mt-1 min-h-[100px] border-green-300 focus:border-green-500 focus:ring-green-500"
            rows={4}
          />
        </div>
      </div>
    </Card>
  );

  const handleSubmit = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await axios.post('http://34.87.113.245:8000/api/text-classification', {
        texts: inputText
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      });
      
      // Process the response based on the expected format
      const data = response.data;
      if (Array.isArray(data) && data.length > 0) {
        const topResult = data.reduce((prev: any, current: any) => 
          (prev.score > current.score) ? prev : current
        );
        setResult({
          predicted_class: topResult.label,
          probabilities: data.sort((a: any, b: any) => b.score - a.score)
        });
      } else {
        setResult(data);
      }
    } catch (err: any) {
      console.error('API Error:', err);
      setError(err.response?.data?.message || err.message || 'An error occurred while processing your request.');
    } finally {
      setLoading(false);
    }
  };

  const ClassificationOutput = () => (
    <Card className="p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Results</h3>
      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
          <span className="ml-2 text-gray-600">Analyzing...</span>
        </div>
      )}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}
      {result && (
        <div className="space-y-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 className="font-medium text-green-800 mb-2">Predicted Class</h4>
            <p className="text-lg font-semibold text-green-900">{result.predicted_class}</p>
          </div>
          {result.probabilities && (
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Confidence Scores</h4>
              <div className="space-y-2">
                {result.probabilities.map((item: any, index: number) => (
                  <div key={index} className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{item.label}</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${(item.score * 100)}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-900">
                        {(item.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );

  return (
    <div className="min-h-screen bg-green-50">
      {/* Header */}
      <header className="bg-green-500 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">Text classification</h1>
          <p className="mt-2 text-green-100 max-w-4xl">
            Given a text passage authored by a person, this task aims to identify the underlying emotion expressed in the text. 
The model classifies the emotion into one of the following categories: anger, disgu...
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-gray-900">Input</h2>
            <TextInputSection />
            <Button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full bg-green-500 hover:bg-green-600 text-white font-medium py-3 px-6 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Processing...' : 'Analyze'}
            </Button>
          </div>

          {/* Output Section */}
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-gray-900">Results</h2>
            <ClassificationOutput />
          </div>
        </div>

        {/* Model Information */}
        <div className="mt-12">
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Model:</span>
                <span className="ml-2 text-gray-600">j-hartmann/emotion-english-distilroberta-base</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">API Endpoint:</span>
                <span className="ml-2 text-gray-600 break-all">http://34.87.113.245:8000/api/text-classification</span>
              </div>
            </div>
            <div className="mt-4">
              <span className="font-medium text-gray-700">Description:</span>
              <p className="mt-1 text-gray-600 text-sm leading-relaxed">
                The model was trained on 6 diverse datasets and predicts Ekman's 6 basic emotions, plus a neutral class....
              </p>
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default App;