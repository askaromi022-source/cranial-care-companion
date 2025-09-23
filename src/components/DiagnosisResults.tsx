import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Brain, Target, Activity, Download, Share } from "lucide-react";

interface DiagnosisData {
  tumorDetected: boolean;
  confidence: number;
  tumorType: string;
  tumorVolume: number;
  location: string;
  segmentationMetrics: {
    diceScore: number;
    hausdorffDistance: number;
    volumeError: number;
  };
}

interface DiagnosisResultsProps {
  results: DiagnosisData;
  isLoading?: boolean;
}

export const DiagnosisResults = ({ results, isLoading = false }: DiagnosisResultsProps) => {
  if (isLoading) {
    return (
      <Card className="shadow-elevated bg-gradient-surface">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary animate-pulse-slow" />
            AI Analysis in Progress
          </CardTitle>
          <CardDescription>NeuroAI Pro is analyzing your MRI scans...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
              <div className="h-4 bg-muted rounded w-1/2"></div>
            </div>
            <Progress value={65} className="animate-pulse-slow" />
            <p className="text-sm text-muted-foreground">
              Processing 3D neural networks and attention mechanisms...
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return "text-green-600";
    if (confidence >= 70) return "text-yellow-600";
    return "text-red-600";
  };

  const getTumorTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case "glioblastoma": return "bg-red-100 text-red-800 border-red-200";
      case "meningioma": return "bg-blue-100 text-blue-800 border-blue-200";
      case "astrocytoma": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Diagnosis Card */}
      <Card className={`shadow-elevated ${results.tumorDetected ? 'border-red-200' : 'border-green-200'}`}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Target className="h-6 w-6 text-primary" />
            Diagnosis Results
          </CardTitle>
          <CardDescription>AI-powered brain tumor detection analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">Tumor Detection</label>
                <div className="flex items-center gap-3 mt-1">
                  <Badge 
                    variant={results.tumorDetected ? "destructive" : "secondary"}
                    className="text-sm px-3 py-1"
                  >
                    {results.tumorDetected ? "TUMOR DETECTED" : "NO TUMOR DETECTED"}
                  </Badge>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-muted-foreground">Confidence Score</label>
                <div className="mt-2">
                  <div className="flex items-center gap-3">
                    <Progress value={results.confidence} className="flex-1" />
                    <span className={`text-lg font-bold ${getConfidenceColor(results.confidence)}`}>
                      {results.confidence}%
                    </span>
                  </div>
                </div>
              </div>

              {results.tumorDetected && (
                <>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Tumor Type</label>
                    <div className="mt-1">
                      <Badge className={getTumorTypeColor(results.tumorType)}>
                        {results.tumorType}
                      </Badge>
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Location</label>
                    <p className="text-sm mt-1">{results.location}</p>
                  </div>
                </>
              )}
            </div>

            {results.tumorDetected && (
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-muted-foreground">Tumor Volume</label>
                  <p className="text-2xl font-bold text-primary mt-1">
                    {results.tumorVolume.toFixed(2)} cm³
                  </p>
                </div>

                <div className="space-y-3">
                  <h4 className="text-sm font-semibold flex items-center gap-2">
                    <Activity className="h-4 w-4" />
                    Segmentation Quality
                  </h4>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Dice Score:</span>
                      <span className="font-mono">{results.segmentationMetrics.diceScore.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Hausdorff Distance:</span>
                      <span className="font-mono">{results.segmentationMetrics.hausdorffDistance.toFixed(1)} mm</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Volume Error:</span>
                      <span className="font-mono">{(results.segmentationMetrics.volumeError * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-3 mt-6 pt-6 border-t">
            <Button variant="medical" className="flex-1">
              <Download className="h-4 w-4" />
              Download Report
            </Button>
            <Button variant="outline" className="flex-1">
              <Share className="h-4 w-4" />
              Share Results
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};