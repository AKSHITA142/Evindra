"use client";

import { use, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useDataset } from "@/hooks/useResearch";
import { DatasetOverview } from "@/components/visuals/DatasetOverview";
import { Spinner } from "@/components/loading/Loading";
import { ErrorState } from "@/components/feedback/ErrorState";
import { Button } from "@/components/buttons/Button";
import { ArrowLeft, Play } from "lucide-react";

export default function OverviewPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { datasetId } = use(params);
  const router = useRouter();

  const { data: dataset, isLoading, error, refetch } = useDataset(datasetId);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <Spinner size="lg" />
        <p className="text-sm text-text-secondary mt-4">Loading dataset profile and schema statistics…</p>
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-12">
        <ErrorState
          title="Dataset Not Found"
          description={`Unable to retrieve profile for dataset ID "${datasetId}".`}
          onRetry={() => refetch()}
          action={
            <Button variant="primary" size="sm" onClick={() => router.push("/upload")}>
              Upload New Dataset
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Top Header Navigation */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" icon={<ArrowLeft className="w-4 h-4" />} onClick={() => router.back()}>
            Back
          </Button>
          <div>
            <h1 className="text-xl font-bold text-text">{dataset.filename}</h1>
            <p className="text-xs text-text-secondary font-mono">Dataset ID: {dataset.dataset_id}</p>
          </div>
        </div>

        <Button
          variant="primary"
          size="sm"
          icon={<Play className="w-4 h-4" />}
          onClick={() => router.push("/upload")}
        >
          New Dataset Run
        </Button>
      </div>

      {/* Live DatasetOverview Component */}
      <DatasetOverview dataset={dataset} />
    </div>
  );
}
