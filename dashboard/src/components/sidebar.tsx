"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  AlertTriangle,
  TrendingUp,
  Zap,
  LineChart,
} from "lucide-react";
import { AGENT_MODEL_LABEL, EMBEDDING_LABEL } from "@/lib/models";

const NAV = [
  { href: "/", label: "Segmentation", icon: BarChart3 },
  { href: "/churn", label: "Churn Risk", icon: AlertTriangle },
  { href: "/uplift", label: "Uplift Intelligence", icon: TrendingUp },
  { href: "/retention", label: "Retention Actions", icon: Zap },
  { href: "/analytics", label: "Audit & Analytics", icon: LineChart },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="flex flex-col w-64 shrink-0 h-full overflow-y-auto"
      style={{
        background: "linear-gradient(175deg, #312E81 0%, #4F46E5 55%, #7C3AED 100%)",
        boxShadow: "6px 0 30px rgba(79,70,229,0.22)",
      }}
    >
      {/* Brand */}
      <div className="px-5 py-6 border-b border-white/15">
        <div className="flex items-center gap-3">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="sg1" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#A78BFA" />
                <stop offset="100%" stopColor="#6366F1" />
              </linearGradient>
            </defs>
            <rect width="40" height="40" rx="11" fill="url(#sg1)" />
            <rect x="9" y="26" width="5" height="8" rx="2" fill="white" opacity="0.9" />
            <rect x="17.5" y="18" width="5" height="16" rx="2" fill="white" />
            <rect x="26" y="11" width="5" height="23" rx="2" fill="white" opacity="0.75" />
            <path d="M11.5 22 L20 15 L28.5 8" stroke="#FCD34D" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="28.5" cy="8" r="3" fill="#FCD34D" />
          </svg>
          <div>
            <div className="text-white font-bold text-[15px] leading-tight tracking-tight">Subscription Churn</div>
            <div className="text-[#C4B5FD] text-[10px] font-semibold uppercase tracking-widest mt-0.5">
              Decision Intelligence
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-white/40 px-3 pb-2">
          Dashboard
        </p>
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-[10px] text-[14px] font-medium transition-all ${
                active
                  ? "bg-white/22 text-white font-semibold"
                  : "text-[#C4B5FD] hover:bg-white/12 hover:text-white"
              }`}
            >
              <Icon size={17} className={active ? "text-white" : "text-[#A78BFA]"} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-white/15">
        <p className="text-[11px] text-white/35 leading-snug">
          CatBoost · CausalML · {EMBEDDING_LABEL}<br />
          {AGENT_MODEL_LABEL}<br />
          <span className="text-white/50">Built by Shivani Bokka</span>
        </p>
      </div>
    </aside>
  );
}
