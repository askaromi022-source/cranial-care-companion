import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { 
  Stethoscope, 
  Calendar, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  UserCheck,
  FileText,
  Phone
} from "lucide-react";

interface Treatment {
  name: string;
  priority: "high" | "medium" | "low";
  description: string;
  timeline: string;
  specialist?: string;
}

interface TreatmentRecommendationsProps {
  tumorType: string;
  tumorVolume: number;
  location: string;
  confidence: number;
}

export const TreatmentRecommendations = ({ 
  tumorType, 
  tumorVolume, 
  location, 
  confidence 
}: TreatmentRecommendationsProps) => {
  
  const getTreatments = (): Treatment[] => {
    const baseLocation = location.toLowerCase();
    const volume = tumorVolume;
    
    switch (tumorType.toLowerCase()) {
      case "glioblastoma":
        return [
          {
            name: "Neurosurgical Consultation",
            priority: "high",
            description: "Immediate evaluation for surgical resection. Time-sensitive intervention required.",
            timeline: "Within 48-72 hours",
            specialist: "Neurosurgeon"
          },
          {
            name: "Radiation Therapy Planning",
            priority: "high", 
            description: "Concurrent chemoradiotherapy with temozolomide following surgical intervention.",
            timeline: "Within 2-4 weeks post-surgery",
            specialist: "Radiation Oncologist"
          },
          {
            name: "Medical Oncology",
            priority: "medium",
            description: "Adjuvant chemotherapy protocol and molecular testing for targeted therapy options.",
            timeline: "Within 4-6 weeks",
            specialist: "Medical Oncologist"
          }
        ];
      
      case "meningioma":
        return [
          {
            name: "Neurosurgical Assessment", 
            priority: volume > 3 ? "high" : "medium",
            description: `${volume > 3 ? "Symptomatic tumor requiring" : "Evaluation for"} surgical resection based on size and location.`,
            timeline: volume > 3 ? "Within 1-2 weeks" : "Within 4-6 weeks",
            specialist: "Neurosurgeon"
          },
          {
            name: "Gamma Knife Radiosurgery",
            priority: "medium",
            description: "Consider for small, deep-seated, or residual tumors post-surgical resection.",
            timeline: "4-8 weeks if indicated",
            specialist: "Radiation Oncologist"
          },
          {
            name: "Regular Monitoring",
            priority: "low",
            description: "Serial MRI imaging for tumor growth assessment if conservative management chosen.",
            timeline: "Every 6-12 months",
            specialist: "Neurologist"
          }
        ];
        
      case "astrocytoma":
        return [
          {
            name: "Neurosurgical Consultation",
            priority: "medium",
            description: "Maximal safe resection with intraoperative neuromonitoring to preserve function.",
            timeline: "Within 2-4 weeks", 
            specialist: "Neurosurgeon"
          },
          {
            name: "Molecular Testing",
            priority: "high",
            description: "IDH mutation, 1p/19q codeletion, and MGMT promoter methylation analysis.",
            timeline: "Immediate (tissue sampling)",
            specialist: "Pathologist"
          },
          {
            name: "Adjuvant Therapy Planning",
            priority: "medium",
            description: "Radiation ± chemotherapy based on molecular profile and tumor grade.",
            timeline: "4-6 weeks post-surgery",
            specialist: "Oncologist"
          }
        ];
        
      default:
        return [
          {
            name: "Specialist Referral",
            priority: "high",
            description: "Immediate referral to neuro-oncology team for comprehensive evaluation.",
            timeline: "Within 24-48 hours",
            specialist: "Neuro-oncologist"
          }
        ];
    }
  };

  const treatments = getTreatments();

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high": return "bg-red-100 text-red-800 border-red-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case "high": return <AlertTriangle className="h-4 w-4" />;
      case "medium": return <Clock className="h-4 w-4" />;
      case "low": return <CheckCircle className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6 animate-slide-up">
      <Card className="shadow-elevated bg-gradient-surface">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Stethoscope className="h-6 w-6 text-primary" />
            Treatment Recommendations
          </CardTitle>
          <CardDescription>
            AI-generated treatment plan based on tumor characteristics and clinical protocols
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Tumor Summary */}
          <div className="bg-accent/30 p-4 rounded-lg mb-6">
            <h4 className="font-semibold mb-2">Case Summary</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Type:</span>
                <p className="font-medium">{tumorType}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Volume:</span>
                <p className="font-medium">{tumorVolume.toFixed(1)} cm³</p>
              </div>
              <div>
                <span className="text-muted-foreground">Location:</span>
                <p className="font-medium">{location}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Confidence:</span>
                <p className="font-medium">{confidence}%</p>
              </div>
            </div>
          </div>

          {/* Treatment Timeline */}
          <div className="space-y-4">
            <h4 className="font-semibold flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              Recommended Treatment Timeline
            </h4>
            
            {treatments.map((treatment, index) => (
              <div key={index} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <Badge className={getPriorityColor(treatment.priority)}>
                        {getPriorityIcon(treatment.priority)}
                        {treatment.priority.toUpperCase()} PRIORITY
                      </Badge>
                      {treatment.specialist && (
                        <Badge variant="outline" className="text-xs">
                          <UserCheck className="h-3 w-3 mr-1" />
                          {treatment.specialist}
                        </Badge>
                      )}
                    </div>
                    <h5 className="font-semibold text-lg">{treatment.name}</h5>
                  </div>
                  <div className="text-right text-sm">
                    <div className="text-muted-foreground">Timeline</div>
                    <div className="font-medium">{treatment.timeline}</div>
                  </div>
                </div>
                
                <p className="text-muted-foreground mb-3">{treatment.description}</p>
                
                {index < treatments.length - 1 && <Separator className="mt-4" />}
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 mt-6 pt-6 border-t">
            <Button variant="medical" className="flex-1">
              <Phone className="h-4 w-4" />
              Schedule Consultation
            </Button>
            <Button variant="success" className="flex-1">
              <FileText className="h-4 w-4" />
              Generate Treatment Plan
            </Button>
          </div>

          {/* Disclaimer */}
          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="font-semibold text-yellow-800 mb-1">Medical Disclaimer</p>
                <p className="text-yellow-700">
                  These recommendations are AI-generated based on imaging analysis and should not replace 
                  professional medical judgment. Always consult with qualified healthcare providers for 
                  diagnosis and treatment decisions.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};