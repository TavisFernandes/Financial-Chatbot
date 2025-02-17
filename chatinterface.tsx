import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, TrendingUp, Building2, Bitcoin, Brain, Leaf } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { ChatMessage, InvestmentOption, UserPreferences } from '../types/investment';
import { analyzeInvestment } from '../data/marketData';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
};

const investmentCategories = [
  {
    id: 'stocks',
    name: 'Stock Market',
    icon: TrendingUp,
    description: 'Tech, growth, and dividend stocks'
  },
  {
    id: 'real-estate',
    name: 'Real Estate',
    icon: Building2,
    description: 'Commercial and residential properties'
  },
  {
    id: 'crypto',
    name: 'Cryptocurrency',
    icon: Bitcoin,
    description: 'Major cryptocurrencies and tokens'
  },
  {
    id: 'ai',
    name: 'AI Technology',
    icon: Brain,
    description: 'AI and automation companies'
  },
  {
    id: 'green',
    name: 'Green Technology',
    icon: Leaf,
    description: 'Sustainable and clean energy'
  }
];

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      text: "Hello! I'm your AI investment advisor. Let's start by understanding your investment goals. How much would you like to invest?",
      isBot: true
    }
  ]);
  const [input, setInput] = useState('');
  const [userPrefs, setUserPrefs] = useState<Partial<UserPreferences>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const createReturnChart = (option: InvestmentOption) => {
    // Simulate historical data based on expected return
    const baseReturn = option.expectedReturn;
    const data = Array(6).fill(0).map((_, i) => {
      const variance = (Math.random() - 0.5) * 5;
      return baseReturn + variance;
    });

    return {
      labels: ['1y', '2y', '3y', '4y', '5y', 'Projected'],
      datasets: [
        {
          label: 'Historical Returns (%)',
          data,
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
        }
      ]
    };
  };

  const handleUserInput = (text: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      text,
      isBot: false
    };
    setMessages(prev => [...prev, userMessage]);

    if (!userPrefs.amount) {
      const amount = parseFloat(text.replace(/[^0-9.]/g, ''));
      if (amount > 0) {
        setUserPrefs(prev => ({ ...prev, amount }));
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: `Great! You're looking to invest ${formatCurrency(amount)}. What's your investment timeframe in years?`,
          isBot: true
        };
        setMessages(prev => [...prev, botResponse]);
      } else {
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: "I couldn't understand the investment amount. Please provide a number (e.g., 10000 or $10,000).",
          isBot: true
        };
        setMessages(prev => [...prev, botResponse]);
      }
    } else if (!userPrefs.timeframe) {
      const timeframe = parseFloat(text.replace(/[^0-9.]/g, ''));
      if (timeframe > 0) {
        setUserPrefs(prev => ({ ...prev, timeframe }));
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: `Perfect! For a ${timeframe}-year investment horizon, what's your desired annual return percentage?`,
          isBot: true
        };
        setMessages(prev => [...prev, botResponse]);
      } else {
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: "Please provide a valid number of years (e.g., 5).",
          isBot: true
        };
        setMessages(prev => [...prev, botResponse]);
      }
    } else if (!userPrefs.desiredReturn) {
      const desiredReturn = parseFloat(text.replace(/[^0-9.]/g, ''));
      if (desiredReturn > 0) {
        setUserPrefs(prev => ({ ...prev, desiredReturn }));
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: "What's your risk tolerance? Please choose: Low, Medium, or High",
          isBot: true
        };
        setMessages(prev => [...prev, botResponse]);
      } else {
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: "Please provide a valid return percentage (e.g., 10).",
          isBot: true
        };
        setMessages(prev => [...prev, botResponse]);
      }
    } else if (!userPrefs.riskTolerance) {
      const riskLevel = text.toLowerCase();
      if (['low', 'medium', 'high'].includes(riskLevel)) {
        setUserPrefs(prev => ({ 
          ...prev, 
          riskTolerance: riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1) as 'Low' | 'Medium' | 'High'
        }));
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: "Great! Now, please select investment sectors you're interested in:",
          isBot: true,
          options: investmentCategories
        };
        setMessages(prev => [...prev, botResponse]);
      } else {
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: "Please choose one of: Low, Medium, or High",
          isBot: true
        };
        setMessages(prev => [...prev, botResponse]);
      }
    } else {
      const category = investmentCategories.find(c => 
        c.name.toLowerCase().includes(text.toLowerCase())
      );
      
      if (category) {
        const recommendations = analyzeInvestment({
          amount: userPrefs.amount!,
          timeframe: userPrefs.timeframe!,
          desiredReturn: userPrefs.desiredReturn!,
          riskTolerance: userPrefs.riskTolerance!,
          sectors: [category.id]
        });

        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: `Based on your preferences (${formatCurrency(userPrefs.amount!)} investment, ${userPrefs.timeframe} years, ${userPrefs.desiredReturn}% desired return, ${userPrefs.riskTolerance} risk), here are your top recommendations for ${category.name}:`,
          isBot: true,
          recommendations,
          charts: recommendations.map(rec => ({
            type: 'line',
            data: createReturnChart(rec)
          }))
        };
        setMessages(prev => [...prev, botResponse]);
        
        // Reset for new conversation
        setUserPrefs({});
      } else {
        const botResponse: ChatMessage = {
          id: (Date.now() + 1).toString(),
          text: "I couldn't recognize that investment category. Please choose from the options above.",
          isBot: true,
          options: investmentCategories
        };
        setMessages(prev => [...prev, botResponse]);
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    handleUserInput(input.trim());
    setInput('');
  };

  return (
    <div className="h-[600px] bg-white rounded-xl shadow-lg flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.isBot ? 'justify-start' : 'justify-end'}`}
          >
            <div
              className={`max-w-[80%] p-4 rounded-lg ${
                message.isBot
                  ? 'bg-gray-100 text-gray-800'
                  : 'bg-blue-600 text-white'
              }`}
            >
              {message.isBot && (
                <Bot className="inline-block w-4 h-4 mr-2 mb-1" />
              )}
              <div className="space-y-4">
                <p>{message.text}</p>
                
                {message.options && (
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    {message.options.map((option) => (
                      <button
                        key={option.id}
                        onClick={() => {
                          setInput(option.name);
                          handleUserInput(option.name);
                        }}
                        className="text-left p-2 text-sm bg-white rounded border border-gray-200 hover:bg-gray-50 transition-colors"
                      >
                        <option.icon className="w-4 h-4 inline-block mr-2" />
                        {option.name}
                      </button>
                    ))}
                  </div>
                )}

                {message.recommendations && (
                  <div className="space-y-4 mt-4">
                    {message.recommendations.map((rec, idx) => (
                      <div key={rec.name} className="bg-white rounded-lg p-4 shadow">
                        <h3 className="font-semibold text-lg mb-2">{rec.name}</h3>
                        <p className="text-gray-600 mb-2">{rec.type}</p>
                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div>
                            <p className="text-sm text-gray-500">Expected Return</p>
                            <p className="font-semibold text-green-600">{rec.expectedReturn}%</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-500">Risk Level</p>
                            <p className="font-semibold">{rec.riskLevel}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-500">Min Investment</p>
                            <p className="font-semibold">{formatCurrency(rec.minInvestment)}</p>
                          </div>
                          {rec.ticker && (
                            <div>
                              <p className="text-sm text-gray-500">Ticker</p>
                              <p className="font-semibold">{rec.ticker}</p>
                            </div>
                          )}
                        </div>
                        {message.charts && message.charts[idx] && (
                          <div className="h-40 mt-4">
                            <Line
                              data={message.charts[idx].data}
                              options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                  legend: {
                                    position: 'top' as const,
                                  },
                                },
                              }}
                            />
                          </div>
                        )}
                        <div className="mt-4">
                          <h4 className="font-semibold mb-2">Key Insights:</h4>
                          <ul className="list-disc list-inside text-sm text-gray-600">
                            {rec.insights.map((insight, i) => (
                              <li key={i}>{insight}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
}