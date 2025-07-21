import React, { useState } from 'react';
import { FileUpload } from '@/components/FileUpload';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, Upload, Zap } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface AccidentalTransaction {
  timestamp: string;
  customer_id: string;
  amount: number;
  recipient_entity: string;
  city: string;
  accident_probability: number;
}

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [results, setResults] = useState<AccidentalTransaction[] | null>(null);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const { toast } = useToast();

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResults(null);
    setHasAnalyzed(false);
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setResults(null);
    setHasAnalyzed(false);
  };

  const analyzeFile = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    
    // Check if we can reach the backend or use demo mode
    const USE_DEMO_MODE = false; // Set to true to test without backend
    
    if (USE_DEMO_MODE) {
      // Demo data for testing the interface
      setTimeout(() => {
        const demoResults = [
          {
            timestamp: "2024-01-15T14:30:00",
            customer_id: "CUST_001",
            amount: 15000,
            recipient_entity: "John's Electronics",
            city: "New York",
            accident_probability: 0.89
          },
          {
            timestamp: "2024-01-15T16:45:00",
            customer_id: "CUST_002", 
            amount: 2500,
            recipient_entity: "Coffee Shop ABC",
            city: "Los Angeles",
            accident_probability: 0.72
          },
          {
            timestamp: "2024-01-15T18:20:00",
            customer_id: "CUST_003",
            amount: 500,
            recipient_entity: "Local Grocery",
            city: "Chicago", 
            accident_probability: 0.45
          }
        ];
        
        setResults(demoResults);
        setHasAnalyzed(true);
        setIsUploading(false);
        
        toast({
          title: "Demo Analysis Complete",
          description: `Found ${demoResults.length} potentially accidental transactions (Demo Mode).`,
        });
      }, 2000);
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('https://flask-accident-api.onrender.com/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze file');
      }

      const data = await response.json();
      setResults(data.accidental_transactions);
      setHasAnalyzed(true);
      
      toast({
        title: "Analysis Complete",
        description: `Found ${data.accidental_transactions.length} potentially accidental transactions.`,
      });
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "Backend Connection Failed", 
        description: "Cannot connect to local Flask server. Make sure it's running on localhost:5000 or enable demo mode.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <AlertTriangle className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Transaction Accident Detector</h1>
              <p className="text-muted-foreground">
                AI-powered analysis to identify potentially accidental transactions
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="space-y-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Upload className="h-5 w-5" />
                  <span>Upload Transaction Data</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <FileUpload
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  onClearFile={handleClearFile}
                  isUploading={isUploading}
                />
                
                {selectedFile && (
                  <div className="flex justify-center">
                    <Button
                      onClick={analyzeFile}
                      disabled={isUploading}
                      size="lg"
                      className="min-w-[200px]"
                    >
                      {isUploading ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent mr-2" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Zap className="h-4 w-4 mr-2" />
                          Analyze Transactions
                        </>
                      )}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="bg-primary/5 border-primary/20">
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-primary/10 rounded">
                      <Zap className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold">AI-Powered</h3>
                      <p className="text-sm text-muted-foreground">
                        Advanced machine learning algorithms detect patterns
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-success/5 border-success/20">
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-success/10 rounded">
                      <AlertTriangle className="h-5 w-5 text-success" />
                    </div>
                    <div>
                      <h3 className="font-semibold">Risk Assessment</h3>
                      <p className="text-sm text-muted-foreground">
                        Probability scoring for each transaction
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-warning/5 border-warning/20">
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-warning/10 rounded">
                      <Upload className="h-5 w-5 text-warning" />
                    </div>
                    <div>
                      <h3 className="font-semibold">CSV Compatible</h3>
                      <p className="text-sm text-muted-foreground">
                        Upload your transaction data in CSV format
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Results Section */}
          {hasAnalyzed && results && (
            <div className="animate-fade-in">
              <ResultsDisplay transactions={results} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;