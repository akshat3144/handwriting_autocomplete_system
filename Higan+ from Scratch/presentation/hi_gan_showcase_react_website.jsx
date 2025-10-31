import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { motion } from 'framer-motion';

const comparisonData = [
  { metric: 'CER (%)', Original: 15.2, Enhanced: 10.3 },
  { metric: 'WER (%)', Original: 42.1, Enhanced: 29.7 },
  { metric: 'FID Score', Original: 68.4, Enhanced: 52.1 },
  { metric: 'Style Fidelity', Original: 72.3, Enhanced: 86.5 }
];

const radarData = [
  { metric: 'Readability', Original: 65, Enhanced: 89 },
  { metric: 'Style', Original: 70, Enhanced: 87 },
  { metric: 'Quality', Original: 62, Enhanced: 82 },
  { metric: 'Diversity', Original: 68, Enhanced: 85 },
  { metric: 'Consistency', Original: 71, Enhanced: 88 }
];
