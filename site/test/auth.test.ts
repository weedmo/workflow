// 관리자 로그인 테스트. 외부(터널) 요청처럼 x-forwarded-for를 붙여 로컬 신뢰를 끄고 확인한다.
import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { makeWorkspace, startServer, type TestServer } from "./helpers.ts";

const ws = makeWorkspace();
let srv: TestServer;
const PW = "test-password-123";
before(async () => { srv = await startServer(ws, { SITE_ADMIN_PASSWORD: PW, SESSION_SECRET: "s".repeat(64) }); });
after(async () => { await srv?.stop(); ws.cleanup(); });

const remote = (ip: string, cookie?: string) => ({ "x-forwarded-for": ip, "content-type": "application/json", ...(cookie ? { cookie } : {}) });
const reindex = (h: Record<string, string>) => fetch(`${srv.url}/api/reindex`, { method: "POST", headers: h, body: "{}" });

test("AUTH-01 외부 요청: 읽기는 열려 있고 쓰기는 401", async () => {
  assert.equal((await fetch(`${srv.url}/api/tree`, { headers: remote("1.2.3.4") })).status, 200);
  assert.equal((await fetch(`${srv.url}/api/search?q=a`, { headers: remote("1.2.3.4") })).status, 200);
  assert.equal((await reindex(remote("1.2.3.4"))).status, 401);
  assert.equal((await fetch(`${srv.url}/api/menus/x`, { method: "DELETE", headers: remote("1.2.3.4") })).status, 401);
  assert.equal((await fetch(`${srv.url}/api/trash`, { headers: remote("1.2.3.4") })).status, 401);
  assert.deepEqual(await (await fetch(`${srv.url}/api/auth/me`, { headers: remote("1.2.3.4") })).json(), { admin: false, local: false });
});

test("AUTH-02 로그인: 틀린 비밀번호 401, 맞으면 쿠키로 쓰기 200, 로그아웃", async () => {
  const bad = await fetch(`${srv.url}/api/auth/login`, { method: "POST", headers: remote("5.6.7.8"), body: JSON.stringify({ password: "nope" }) });
  assert.equal(bad.status, 401);
  const ok = await fetch(`${srv.url}/api/auth/login`, { method: "POST", headers: { ...remote("5.6.7.8"), "x-forwarded-proto": "https" }, body: JSON.stringify({ password: PW }) });
  assert.equal(ok.status, 200);
  const set = ok.headers.get("set-cookie")!;
  assert.match(set, /HttpOnly/); assert.match(set, /SameSite=Lax/); assert.match(set, /Secure/);
  const cookie = set.split(";")[0];
  assert.equal((await reindex(remote("5.6.7.8", cookie))).status, 200);
  assert.equal((await reindex(remote("5.6.7.8", cookie.slice(0, -2) + "xx"))).status, 401, "변조한 쿠키");
});

test("AUTH-03 로그인 시도 제한: 10번 틀리면 429", async () => {
  let last = 0;
  for (let i = 0; i < 11; i++) last = (await fetch(`${srv.url}/api/auth/login`, { method: "POST", headers: remote("9.9.9.9"), body: JSON.stringify({ password: "x" }) })).status;
  assert.equal(last, 429);
});

test("AUTH-04 로컬(프록시 헤더 없음) 요청은 로그인 없이 쓸 수 있다(MCP)", async () => {
  assert.equal((await reindex({ "content-type": "application/json" })).status, 200);
});
