import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HttpError, useTree } from "./lib/api.ts";
import { ConfirmProvider } from "./components/confirm/ConfirmProvider.tsx";
import { Layout } from "./components/Layout.tsx";
import { DocView } from "./components/DocView.tsx";
import { EmptyState } from "./components/EmptyState.tsx";
import { TooltipLayer } from "./components/Tooltip.tsx";
import { loadAuth } from "./lib/auth.ts";
import "./styles.css";

// 개발 모드에서만: 화면 요소를 가리키고 ⌘C로 복사하면 컴포넌트·파일 위치를 코딩 에이전트에 붙여 넣을 수 있다 (react-grab.com)
if (import.meta.env.DEV) void import("react-grab");

/** 메인(/): 첫 대분류로 보낸다 */
function Home() {
  const { data, error } = useTree();
  if (error) return <EmptyState title="서버에 연결할 수 없습니다" body={String(error)} />;
  const first = data?.categories[0]?.key;
  return first ? <Navigate to={`/${first}`} replace /> : <EmptyState title="불러오는 중…" />;
}

const router = createBrowserRouter([
  { path: "/", element: <Home /> },
  {
    path: "/:category",
    element: <Layout />,
    children: [
      { index: true, element: <EmptyState title="문서를 고르세요" body="왼쪽 목록에서 문서를 선택하면 여기에서 열립니다." /> },
      { path: ":menu/:doc", element: <DocView /> },
    ],
  },
]);

// 404(옮겨지거나 지워진 문서)는 다시 시도해도 같으므로 재시도하지 않는다
const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: (n, e) => !(e instanceof HttpError && e.status === 404) && n < 1 } },
});

void loadAuth().then(() => createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ConfirmProvider>
        <RouterProvider router={router} />
      </ConfirmProvider>
      <TooltipLayer />
    </QueryClientProvider>
  </StrictMode>,
));
