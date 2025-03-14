import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

// Main Dashboard Component
const CryptographicRiskDashboard = () => {
  // Sample data - in a real implementation, this would come from APIs or databases
  const [organizationData, setOrganizationData] = useState({
    riskScore: 68,
    complianceStatus: 'Partial Compliance',
    vulnerableAssets: 132,
    totalAssets: 457,
    criticalVulnerabilities: 24,
    lastUpdated: new Date().toLocaleString()
  });

  const [businessImpactData, setBusinessImpactData] = useState([
    { name: 'Financial Services', assets: 112, vulnerableAssets: 42, criticalAssets: 78, riskScore: 76 },
    { name: 'Customer Data', assets: 86, vulnerableAssets: 28, criticalAssets: 64, riskScore: 83 },
    { name: 'Supply Chain', assets: 64, vulnerableAssets: 18, criticalAssets: 31, riskScore: 58 },
    { name: 'HR Systems', assets: 48, vulnerableAssets: 12, criticalAssets: 15, riskScore: 42 },
    { name: 'Email & Comms', assets: 94, vulnerableAssets: 24, criticalAssets: 35, riskScore: 63 },
    { name: 'Research & IP', assets: 53, vulnerableAssets: 8, criticalAssets: 47, riskScore: 72 }
  ]);

  const [riskTrendData, setRiskTrendData] = useState([
    { month: 'Jan', riskScore: 82, industry: 78 },
    { month: 'Feb', riskScore: 79, industry: 77 },
    { month: 'Mar', riskScore: 77, industry: 76 },
    { month: 'Apr', riskScore: 74, industry: 75 },
    { month: 'May', riskScore: 72, industry: 76 },
    { month: 'Jun', riskScore: 68, industry: 75 }
  ]);

  const [vulnerabilityDistribution, setVulnerabilityDistribution] = useState([
    { name: 'Legacy RSA/ECC', value: 42, color: '#FF8042' },
    { name: 'Weak Parameters', value: 28, color: '#FFBB28' },
    { name: 'Implementation Flaws', value: 15, color: '#00C49F' },
    { name: 'Protocol Downgrades', value: 9, color: '#0088FE' },
    { name: 'Certificate Issues', value: 6, color: '#8884d8' }
  ]);

  const [complianceData, setComplianceData] = useState([
    { name: 'NIST 800-53', compliant: 86, gap: 14 },
    { name: 'PCI-DSS', compliant: 92, gap: 8 },
    { name: 'HIPAA', compliant: 78, gap: 22 },
    { name: 'GDPR', compliant: 84, gap: 16 },
    { name: 'ISO 27001', compliant: 88, gap: 12 },
    { name: 'NIST PQC', compliant: 34, gap: 66 }
  ]);

  const [resourceImpactData, setResourceImpactData] = useState([
    { name: 'Current', security: 68, performance: 82, cost: 65, complexity: 45, compatibility: 88 },
    { name: 'Proposed', security: 92, performance: 76, cost: 72, complexity: 58, compatibility: 81 }
  ]);

  // KPI Cards for the header section
  const KpiCard = ({ title, value, description, trend, color }) => {
    return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <h3 className="text-lg font-semibold text-gray-700">{title}</h3>
        <div className="flex items-baseline">
          <span className="text-3xl font-bold" style={{ color }}>{value}</span>
          {trend && (
            <span className={`ml-2 ${trend > 0 ? 'text-red-500' : 'text-green-500'}`}>
              {trend > 0 ? `▲ ${trend}%` : `▼ ${Math.abs(trend)}%`}
            </span>
          )}
        </div>
        <p className="mt-1 text-sm text-gray-500">{description}</p>
      </div>
    );
  };

  // Heat Map for Business Impact Visualization
  const BusinessImpactHeatMap = ({ data }) => {
    return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Business Impact Heat Map</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Business Unit
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Assets
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Vulnerable
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Critical
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Risk Score
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.map((unit, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {unit.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {unit.assets}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {unit.vulnerableAssets} ({Math.round((unit.vulnerableAssets / unit.assets) * 100)}%)
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {unit.criticalAssets}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div 
                        className="h-4 rounded-full" 
                        style={{ 
                          width: `${unit.riskScore}%`, 
                          backgroundColor: unit.riskScore > 80 ? '#ef4444' : unit.riskScore > 60 ? '#f97316' : unit.riskScore > 40 ? '#eab308' : '#22c55e'
                        }}
                      ></div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Risk Trend Analysis
  const RiskTrendAnalysis = ({ data }) => {
    return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Risk Score Trend Analysis</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="riskScore" name="Organization Score" stroke="#8884d8" activeDot={{ r: 8 }} />
            <Line type="monotone" dataKey="industry" name="Industry Average" stroke="#82ca9d" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Vulnerability Distribution
  const VulnerabilityDistributionChart = ({ data }) => {
    return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Vulnerability Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => [`${value} issues`, 'Count']} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Compliance Status
  const ComplianceStatus = ({ data }) => {
    return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Regulatory Compliance Status</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="name" />
            <Tooltip formatter={(value) => [`${value}%`, 'Compliance']} />
            <Legend />
            <Bar dataKey="compliant" stackId="a" fill="#82ca9d" name="Compliant" />
            <Bar dataKey="gap" stackId="a" fill="#ff7675" name="Gap" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Resource Allocation Impact
  const ResourceAllocationImpact = ({ data }) => {
    return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Resource Allocation Impact</h2>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="name" />
            <PolarRadiusAxis angle={30} domain={[0, 100]} />
            <Radar name="Current State" dataKey="security" stroke="#8884d8" fill="#8884d8" fillOpacity={0.2} />
            <Radar name="Proposed Improvements" dataKey="security" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.2} />
            <Tooltip />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
        <div className="mt-4 flex flex-col space-y-2">
          <div className="flex justify-between">
            <span className="text-sm font-medium">Security</span>
            <span className="text-sm text-green-600">+24% improvement</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm font-medium">Performance</span>
            <span className="text-sm text-red-600">-6% impact</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm font-medium">Implementation Cost</span>
            <span className="text-sm">Estimated $1.2M</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm font-medium">ROI (3 year)</span>
            <span className="text-sm text-green-600">215%</span>
          </div>
        </div>
      </div>
    );
  };

  // Action Items Section
  const ActionItems = () => {
    const priorityActions = [
      { id: 1, title: "Replace RSA-1024 on financial transaction servers", impact: "High", effort: "Medium", status: "In Progress" },
      { id: 2, title: "Upgrade TLS configurations organization-wide", impact: "High", effort: "Low", status: "Planned" },
      { id: 3, title: "Implement Kyber-768 for VPN infrastructure", impact: "Medium", effort: "High", status: "Planned" },
      { id: 4, title: "Remediate certificate validation issues", impact: "Medium", effort: "Low", status: "Completed" }
    ];

    return (
      <div className="p-4 bg-white rounded-lg shadow-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Priority Action Items</h2>
          <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Generate Report</button>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Business Impact</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Implementation Effort</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {priorityActions.map((action) => (
                <tr key={action.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{action.title}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      action.impact === 'High' ? 'bg-red-100 text-red-800' : 
                      action.impact === 'Medium' ? 'bg-yellow-100 text-yellow-800' : 
                      'bg-green-100 text-green-800'
                    }`}>
                      {action.impact}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      action.effort === 'High' ? 'bg-red-100 text-red-800' : 
                      action.effort === 'Medium' ? 'bg-yellow-100 text-yellow-800' : 
                      'bg-green-100 text-green-800'
                    }`}>
                      {action.effort}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      action.status === 'Completed' ? 'bg-green-100 text-green-800' : 
                      action.status === 'In Progress' ? 'bg-blue-100 text-blue-800' : 
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {action.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      {/* Header with title and last updated */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Executive Cryptographic Risk Dashboard</h1>
        <div className="text-sm text-gray-500">Last updated: {organizationData.lastUpdated}</div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard 
          title="Organizational Risk Score" 
          value={`${organizationData.riskScore}/100`} 
          description="Overall cryptographic vulnerability exposure" 
          trend={-4}
          color={organizationData.riskScore > 80 ? '#ef4444' : organizationData.riskScore > 60 ? '#f97316' : organizationData.riskScore > 40 ? '#eab308' : '#22c55e'}
        />
        <KpiCard 
          title="Compliance Status" 
          value={organizationData.complianceStatus} 
          description="Current regulatory compliance standing"
          color="#3b82f6" 
        />
        <KpiCard 
          title="Vulnerable Assets" 
          value={`${organizationData.vulnerableAssets}/${organizationData.totalAssets}`} 
          description="Assets with cryptographic vulnerabilities"
          trend={-2}
          color="#f97316"
        />
        <KpiCard 
          title="Critical Vulnerabilities" 
          value={organizationData.criticalVulnerabilities} 
          description="High-impact, high-likelihood issues"
          trend={-5}
          color="#ef4444"
        />
      </div>

      {/* Two column layout for charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <BusinessImpactHeatMap data={businessImpactData} />
        <RiskTrendAnalysis data={riskTrendData} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <VulnerabilityDistributionChart data={vulnerabilityDistribution} />
        <ComplianceStatus data={complianceData} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ResourceAllocationImpact data={resourceImpactData} />
        <ActionItems />
      </div>

      {/* Executive Summary Section */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <h2 className="text-xl font-bold mb-4">Executive Summary</h2>
        <div className="prose max-w-none">
          <p>
            The organization's cryptographic security posture has <strong>improved by 4%</strong> over the past quarter, 
            with significant progress in remediating high-risk vulnerabilities in financial processing systems.
          </p>
          <p>
            <strong>Key findings:</strong> 29% of our digital assets still use cryptographic implementations vulnerable 
            to quantum computing attacks, with the highest concentration in financial services (38%) and customer data systems (33%).
          </p>
          <p>
            <strong>Recommended focus:</strong> The most impactful next step is upgrading TLS configurations organization-wide, 
            which would reduce our risk score by approximately 8 points while requiring relatively low implementation effort.
          </p>
          <p>
            <strong>Long-term outlook:</strong> Based on current quantum computing development trajectories, we recommend 
            a 3-year transition plan to quantum-resistant algorithms, prioritizing customer data and financial systems in year 1.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CryptographicRiskDashboard;