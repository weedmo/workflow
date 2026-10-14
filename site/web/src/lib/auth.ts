/**
 * 로그인 상태. 읽기(목록·문서·검색)는 누구나, 쓰기(업로드·편집·관리)는 관리자 로그인 후에만.
 * 화면을 그리기 전에 loadAuth()로 한 번 읽고, 로그인·로그아웃하면 페이지를 새로 고친다.
 */
export let READONLY = true;

export async function loadAuth() {
  try {
    const r = await fetch("/api/auth/me");
    READONLY = !((await r.json()) as { admin?: boolean }).admin;
  } catch { READONLY = true; }
}

export async function login(password: string): Promise<string | null> {
  const r = await fetch("/api/auth/login", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ password }),
  });
  if (r.ok) { location.reload(); return null; }
  return ((await r.json().catch(() => ({}))) as { error?: string }).error ?? `${r.status} 오류`;
}

export async function logout() {
  await fetch("/api/auth/logout", { method: "POST" });
  location.reload();
}
