import React from 'react';
import { AlertTriangle, Clock, User, MapPin, DollarSign } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface AccidentalTransaction {
  timestamp: string;
  customer_id: string;
  amount: number;
  recipient_entity: string;
  city: string;
  accident_probability: number;
}

interface ResultsDisplayProps {
  transactions: AccidentalTransaction[];
}

const getRiskLevel = (probability: number) => {
  if (probability >= 0.8) return { level: 'High', color: 'destructive' };
  if (probability >= 0.6) return { level: 'Medium', color: 'warning' };
  return { level: 'Low', color: 'secondary' };
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
};

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ transactions }) => {
  if (transactions.length === 0) {
    return (
      <Card className="p-8 text-center">
        <div className="flex flex-col items-center space-y-4">
          <div className="p-4 bg-success/10 rounded-full">
            <AlertTriangle className="h-8 w-8 text-success" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-success mb-2">No Accidental Transactions Detected</h3>
            <p className="text-muted-foreground">
              All transactions appear to be legitimate based on the analysis.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const highRiskCount = transactions.filter(t => t.accident_probability >= 0.8).length;
  const mediumRiskCount = transactions.filter(t => t.accident_probability >= 0.6 && t.accident_probability < 0.8).length;
  const lowRiskCount = transactions.filter(t => t.accident_probability < 0.6).length;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              <div>
                <p className="text-sm font-medium">Total Flagged</p>
                <p className="text-2xl font-bold">{transactions.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 rounded-full bg-destructive"></div>
              <div>
                <p className="text-sm font-medium">High Risk</p>
                <p className="text-2xl font-bold text-destructive">{highRiskCount}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 rounded-full bg-warning"></div>
              <div>
                <p className="text-sm font-medium">Medium Risk</p>
                <p className="text-2xl font-bold text-warning">{mediumRiskCount}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 rounded-full bg-muted-foreground"></div>
              <div>
                <p className="text-sm font-medium">Low Risk</p>
                <p className="text-2xl font-bold text-muted-foreground">{lowRiskCount}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Transactions List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5" />
            <span>Flagged Transactions</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {transactions.map((transaction, index) => {
              const risk = getRiskLevel(transaction.accident_probability);
              return (
                <div
                  key={index}
                  className="flex flex-col lg:flex-row lg:items-center justify-between p-4 border rounded-lg hover:bg-secondary/30 transition-colors"
                >
                  <div className="flex-1 space-y-2 lg:space-y-0">
                    <div className="flex flex-wrap items-center gap-3">
                      <Badge variant={risk.color as any} className="font-medium">
                        {risk.level} Risk
                      </Badge>
                      <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
                        {(transaction.accident_probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                      <div className="flex items-center space-x-2">
                        <DollarSign className="h-4 w-4 text-muted-foreground" />
                        <span className="font-semibold">{formatCurrency(transaction.amount)}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <User className="h-4 w-4 text-muted-foreground" />
                        <span className="truncate">{transaction.recipient_entity}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <MapPin className="h-4 w-4 text-muted-foreground" />
                        <span>{transaction.city}</span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Clock className="h-4 w-4 text-muted-foreground" />
                        <span>{formatDate(transaction.timestamp)}</span>
                      </div>
                    </div>
                    
                    <div className="text-xs text-muted-foreground">
                      Customer ID: {transaction.customer_id}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};