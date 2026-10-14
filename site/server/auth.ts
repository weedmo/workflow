import { createHmac, randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import type { FastifyInstance, FastifyRequest } from "fastify";

/**
 * 관리자 로그인(관리자 한 명).
 * - 읽기(GET: 트리·문서·파일·검색·변경 알림)는 누구나. 쓰기(GET 아닌 /api/*, 그리고 GET /api/trash)는 로그인 필요, 아니면 401.
 * - 비밀번호: SITE_ADMIN_PASSWORD(시작할 때 scrypt 해시로만 메모리에 두고 timingSafeEqual로 비교).
 * - 세션: SESSION_SECRET으로 HMAC 서명한 쿠키(만료 시각 포함, 상태 없음). httpOnly·SameSite=Lax, HTTPS(X-Forwarded-Proto)면 Secure.
 * - 로컬 신뢰: 127.0.0.1/::1에서 왔고 프록시 헤더(cf-connecting-ip·x-forwarded-for)가 없는 요청은 로그인 없이 허용한다
 *   (같은 머신의 MCP·Claude Code). Cloudflare Tunnel(cloudflared)은 localhost에서 붙지만 이 헤더를 붙이므로 외부 요청은 신뢰되지 않는다.
 *   WORKFLOW_TRUST_LOCAL=0이면 로컬도 로그인해야 한다.
 * - SITE_ADMIN_PASSWORD가 없으면 로컬 신뢰 요청만 쓸 수 있다(외부 쓰기는 모두 401).
 */
const COOKIE = "wf_session";
const TTL_MS = 30 * 24 * 3600 * 1000;
const WRITE_FREE = new Set(["/api/auth/login", "/api/auth/logout"]);

export function registerAuth(app: FastifyInstance) {
  const password = process.env.SITE_ADMIN_PASSWORD ?? "";
  const secret = process.env.SESSION_SECRET || randomBytes(32).toString("hex");
  const trustLocal = process.env.WORKFLOW_TRUST_LOCAL !== "0";
  const salt = randomBytes(16);
  const hash = password ? scryptSync(password, salt, 32) : null;

  const sign = (v: string) => createHmac("sha256", secret).update(v).digest("base64url");
  const token = () => { const v = `admin.${Date.now() + TTL_MS}`; return `${v}.${sign(v)}`; };
  const validToken = (t: string | undefined) => {
    if (!t) return false;
    const i = t.lastIndexOf(".");
    const v = t.slice(0, i), sig = Buffer.from(t.slice(i + 1));
    const want = Buffer.from(sign(v));
    if (sig.length !== want.length || !timingSafeEqual(sig, want)) return false;
    return Number(v.split(".")[1]) > Date.now();
  };
  const cookieOf = (req: FastifyRequest) =>
    (req.headers.cookie ?? "").split(/;\s*/).find((c) => c.startsWith(COOKIE + "="))?.slice(COOKIE.length + 1);
  const isLocal = (req: FastifyRequest) => {
    const ip = req.socket.remoteAddress ?? "";
    return trustLocal && ["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(ip)
      && !req.headers["cf-connecting-ip"] && !req.headers["x-forwarded-for"];
  };
  const isAdmin = (req: FastifyRequest) => isLocal(req) || validToken(cookieOf(req));
  const secure = (req: FastifyRequest) => req.headers["x-forwarded-proto"] === "https" || !!req.headers["cf-visitor"]?.toString().includes("https");
  const setCookie = (req: FastifyRequest, value: string, maxAge: number) =>
    `${COOKIE}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAge}${secure(req) ? "; Secure" : ""}`;

  app.addHook("onRequest", async (req, reply) => {
    const path = req.url.split("?")[0];
    if (!path.startsWith("/api/") || WRITE_FREE.has(path)) return;
    const write = !["GET", "HEAD", "OPTIONS"].includes(req.method) || path === "/api/trash";
    if (write && !isAdmin(req)) return reply.code(401).send({ error: "로그인이 필요합니다" });
  });

  // 로그인 시도 제한: IP마다 15분에 10번
  const attempts = new Map<string, { n: number; until: number }>();
  const clientIp = (req: FastifyRequest) => String(req.headers["cf-connecting-ip"] ?? req.headers["x-forwarded-for"] ?? req.ip).split(",")[0].trim();

  app.get("/api/auth/me", async (req) => ({ admin: isAdmin(req), local: isLocal(req) }));
  app.post<{ Body: { password?: string } }>("/api/auth/login", async (req, reply) => {
    const ip = clientIp(req), now = Date.now();
    const a = attempts.get(ip);
    if (a && a.until > now && a.n >= 10) return reply.code(429).send({ error: "시도가 너무 많습니다. 잠시 뒤 다시 하세요" });
    const given = String(req.body?.password ?? "");
    const ok = !!hash && timingSafeEqual(scryptSync(given, salt, 32), hash);
    if (!ok) {
      attempts.set(ip, a && a.until > now ? { n: a.n + 1, until: a.until } : { n: 1, until: now + 15 * 60_000 });
      return reply.code(401).send({ error: hash ? "비밀번호가 틀렸습니다" : "관리자 비밀번호가 설정되지 않았습니다" });
    }
    attempts.delete(ip);
    reply.header("set-cookie", setCookie(req, token(), TTL_MS / 1000));
    return { ok: true };
  });
  app.post("/api/auth/logout", async (req, reply) => {
    reply.header("set-cookie", setCookie(req, "", 0));
    return { ok: true };
  });
}
