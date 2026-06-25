import {
  getUpliftKpis,
  getCustomerTypeSummary,
  getRoiBySegment,
  getTopPersuadables,
  getUpliftScatterData,
} from "@/lib/data";
import { UpliftClient } from "@/components/pages/uplift-client";

export const dynamic = "force-dynamic";

export default async function UpliftPage() {
  const [kpis, typeSummary, roiBySeg, topPersuadables, scatter] = await Promise.all([
    getUpliftKpis(),
    getCustomerTypeSummary(),
    getRoiBySegment(),
    getTopPersuadables(15),
    getUpliftScatterData(),
  ]);
  return (
    <UpliftClient
      kpis={kpis}
      typeSummary={typeSummary}
      roiBySeg={roiBySeg}
      topPersuadables={topPersuadables}
      scatter={scatter}
    />
  );
}
