import { useState } from "react";
import { Modal, ui } from "./Modal.tsx";
import { login } from "../lib/auth.ts";

/** 관리자 로그인. 성공하면 페이지를 새로 고쳐 쓰기 기능을 켠다 */
export function LoginDialog({ onClose }: { onClose: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <Modal title="관리자 로그인" onClose={onClose}>
      <form className="grid gap-3" onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        setError(await login(password));
        setBusy(false);
      }}>
        <label className={ui.label}>비밀번호
          <input className={ui.field} type="password" autoComplete="current-password" autoFocus value={password} onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <p className={ui.error} role="alert">{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className={ui.secondary} onClick={onClose}>취소</button>
          <button type="submit" className={ui.primary} disabled={busy || !password}>로그인</button>
        </div>
      </form>
    </Modal>
  );
}
