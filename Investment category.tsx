import React from 'react';
import { XIcon as Icon } from 'lucide-react';

interface InvestmentCategoryProps {
  title: string;
  description: string;
  icon: Icon;
}

export default function InvestmentCategory({
  title,
  description,
  icon: IconComponent,
}: InvestmentCategoryProps) {
  return (
    <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
      <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
        <IconComponent className="w-6 h-6 text-blue-600" />
      </div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}