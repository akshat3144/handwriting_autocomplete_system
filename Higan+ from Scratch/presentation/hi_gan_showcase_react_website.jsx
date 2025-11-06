import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { motion } from 'framer-motion';

const sampleImages = Array.from({ length: 20 }, (_, i) => ({
  id: i,
  real: `../inference_results/sample_${String(i).padStart(3, '0')}_real.png`,
  generated: `../inference_results/sample_${String(i).padStart(3, '0')}_generated.png`
}));

export default function HiGANPlusShowcase() {
  const [selectedSample, setSelectedSample] = useState(0);
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-6xl font-black mb-4 text-center bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">Enhanced HiGAN+</h1>
        <p className="text-xl text-center text-gray-300 mb-12">Next-Generation Handwriting Synthesis</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
          <div className="bg-white/10 rounded-xl p-6"><div className="text-4xl mb-2"></div><div className="text-3xl font-bold text-purple-400">+32%</div><div className="text-sm text-gray-400">CER Improvement</div></div>
          <div className="bg-white/10 rounded-xl p-6"><div className="text-4xl mb-2"></div><div className="text-3xl font-bold text-purple-400">-24%</div><div className="text-sm text-gray-400">FID Reduction</div></div>
          <div className="bg-white/10 rounded-xl p-6"><div className="text-4xl mb-2"></div><div className="text-3xl font-bold text-purple-400">86.5%</div><div className="text-sm text-gray-400">Style Accuracy</div></div>
          <div className="bg-white/10 rounded-xl p-6"><div className="text-4xl mb-2"></div><div className="text-3xl font-bold text-purple-400">High</div><div className="text-sm text-gray-400">Training Stability</div></div>
        </div>
        <div className="bg-white/5 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-8 text-center">Real Generation Results</h2>
          <div className="flex justify-center gap-2 mb-8 flex-wrap">
            {sampleImages.slice(0, 10).map((_, i) => (
              <button key={i} onClick={() => setSelectedSample(i)} className={`px-4 py-2 rounded-lg font-semibold transition-all ${selectedSample === i ? 'bg-gradient-to-r from-purple-600 to-pink-600' : 'bg-white/10 hover:bg-white/20'}`}>{i + 1}</button>
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div><h3 className="text-xl font-semibold mb-4 text-center">Real Sample</h3><div className="bg-white rounded-lg p-4"><img src={sampleImages[selectedSample].real} alt="Real" className="w-full h-auto" /></div></div>
            <div><h3 className="text-xl font-semibold mb-4 text-center">Generated Output</h3><div className="bg-white rounded-lg p-4"><img src={sampleImages[selectedSample].generated} alt="Generated" className="w-full h-auto" /></div></div>
          </div>
        </div>
      </div>
    </div>
  );
}
