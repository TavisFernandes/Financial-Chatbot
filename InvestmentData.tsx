import { InvestmentCategory } from '../types/investment';
import {
  TrendingUp, Building2, Bitcoin, Brain,
  Leaf, Rocket, BadgeDollarSign, BarChart3,
  ShoppingBag
} from 'lucide-react';

export const investmentCategories: InvestmentCategory[] = [
  {
    id: 'stocks',
    name: 'Stock Market',
    icon: TrendingUp,
    description: 'Diversified portfolio of stocks across sectors',
    options: [
      {
        id: 'tech-growth',
        name: 'Tech Growth Portfolio',
        description: 'High-growth technology companies with strong fundamentals',
        expectedReturn: 15,
        riskLevel: 'High',
        minInvestment: 5000,
        recommendedTimeframe: '5+ years',
        trends: [8, 12, 15, 10, 18, 14],
        insights: [
          'Strong tech sector performance',
          'High innovation potential',
          'Market leadership positions'
        ]
      },
      {
        id: 'dividend-stocks',
        name: 'Dividend Champions',
        description: 'Stable companies with consistent dividend growth',
        expectedReturn: 8,
        riskLevel: 'Low',
        minInvestment: 10000,
        recommendedTimeframe: '3+ years',
        trends: [6, 7, 8, 7.5, 8.2, 8.5],
        insights: [
          'Reliable income stream',
          'Lower volatility',
          'Established businesses'
        ]
      },
      {
        id: 'growth-blend',
        name: 'Growth & Value Blend',
        description: 'Mix of growth and value stocks for balanced returns',
        expectedReturn: 12,
        riskLevel: 'Medium',
        minInvestment: 7500,
        recommendedTimeframe: '4+ years',
        trends: [7, 10, 9, 11, 13, 12],
        insights: [
          'Balanced approach',
          'Sector diversification',
          'Growth potential with stability'
        ]
      }
    ]
  },
  {
    id: 'real-estate',
    name: 'Real Estate',
    icon: Building2,
    description: 'Property investments and REITs',
    options: [
      {
        id: 'commercial-reits',
        name: 'Commercial REITs',
        description: 'Investment in commercial property trusts',
        expectedReturn: 10,
        riskLevel: 'Medium',
        minInvestment: 25000,
        recommendedTimeframe: '5+ years',
        trends: [7, 8, 9, 10, 9.5, 10.2],
        insights: [
          'Stable commercial tenants',
          'Regular income distribution',
          'Property appreciation potential'
        ]
      },
      {
        id: 'residential-development',
        name: 'Residential Development Projects',
        description: 'Investment in residential property developments',
        expectedReturn: 18,
        riskLevel: 'High',
        minInvestment: 50000,
        recommendedTimeframe: '3-5 years',
        trends: [12, 15, 20, 18, 16, 19],
        insights: [
          'High demand areas',
          'Development profit potential',
          'Exit strategy flexibility'
        ]
      },
      {
        id: 'reit-portfolio',
        name: 'Diversified REIT Portfolio',
        description: 'Mix of different real estate investment trusts',
        expectedReturn: 12,
        riskLevel: 'Medium',
        minInvestment: 15000,
        recommendedTimeframe: '4+ years',
        trends: [8, 10, 11, 12, 11.5, 12.5],
        insights: [
          'Sector diversification',
          'Professional management',
          'Liquidity advantages'
        ]
      }
    ]
  },
  // Add more categories with their options...
];

export const analyzeInvestment = (preferences: {
  amount: number;
  timeframe: number;
  desiredReturn: number;
  category?: string;
}) => {
  const { amount, timeframe, desiredReturn, category } = preferences;
  let recommendations: InvestmentOption[] = [];

  // Filter by category if specified
  let options = category
    ? investmentCategories.find(c => c.id === category)?.options || []
    : investmentCategories.flatMap(c => c.options);

  // Filter by minimum investment
  options = options.filter(opt => opt.minInvestment <= amount);

  // Sort by match to desired return and risk level
  options.sort((a, b) => {
    const aMatch = Math.abs(a.expectedReturn - desiredReturn);
    const bMatch = Math.abs(b.expectedReturn - desiredReturn);
    return aMatch - bMatch;
  });

  // Take top 3 recommendations
  recommendations = options.slice(0, 3);

  return recommendations;
};