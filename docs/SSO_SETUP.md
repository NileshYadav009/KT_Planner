# Single sign-on setup

Continuum signs people in with your company's identity provider using OpenID Connect. Each Continuum workspace connects to one provider. People sign in with their work email; machines (CI, scripts) keep using API keys.

## 1. Register Continuum with your identity provider

Create a **web application** (confidential client) with:

| Setting | Value |
|---|---|
| Redirect URI | `https://<your Continuum host>/sso/callback` |
| Grant type | Authorization code (PKCE is used automatically) |
| Scopes | `openid email profile` |

Then note the **issuer URL**, the **client ID** and the **client secret**.

| Provider | Issuer URL | Notes |
|---|---|---|
| Microsoft Entra ID | `https://login.microsoftonline.com/<directory id>/v2.0` | App registrations → New registration → Web platform. To map roles, define app roles (e.g. `Continuum.Admin`) and assign them; they arrive in the `roles` claim. Turn on "Assignment required" to control who can sign in. |
| Okta | `https://<org>.okta.com/oauth2/default` | Applications → Create App Integration → OIDC, Web Application. To map roles from groups, add a `groups` claim to the ID token. |
| Google Workspace | `https://accounts.google.com` | Google Cloud console → Credentials → OAuth client ID → Web application. Only accounts in your Workspace domain are accepted (the `hd` claim is checked). Google sends no groups, so set roles in Continuum. |
| Keycloak | `https://<host>/realms/<realm>` | Client with "Client authentication" on and "Standard flow" enabled. |

## 2. Configure the Continuum server

Put the client secret in an environment variable; Continuum stores only the variable's name:

```
CONTINUUM_PUBLIC_URL=https://kt.example.com      # the address people use; needed behind a proxy
CONTINUUM_SECURE_COOKIES=1                       # if TLS ends at a proxy
ACME_SSO_SECRET=<client secret>
```

Then connect the workspace (the tenant id comes from `python -m auth list`):

```
python -m auth sso-set <tenant_id> \
    --issuer https://login.microsoftonline.com/<directory id>/v2.0 \
    --client-id <application id> \
    --client-secret-env ACME_SSO_SECRET \
    --domains acme.com,acme.co.uk
```

The command checks the issuer's discovery document and prints the redirect URI to register.

| Option | Effect |
|---|---|
| `--domains` | Email domains that sign in to this workspace. A domain can belong to one workspace only. |
| `--default-role receiver` | Role for a person signing in for the first time (default `receiver`, read-only). |
| `--invite-only` | Only people added beforehand with `python -m auth add-user` can sign in. |
| `--role-claim roles --role-map "Continuum.Admin=admin,Continuum.Reviewer=reviewer,Continuum.Giver=giver"` | Take the role from the provider at every sign-in. Someone removed from the group loses the role at their next sign-in. |
| `--enforce` | People must use SSO; an API key can no longer open a browser session (it still works for API calls). |

## 3. Roles

| Role | Can |
|---|---|
| `admin` | Everything, including deleting any KT in the workspace |
| `reviewer` | Create and correct KTs; delete their own |
| `giver` | The person handing over: create and correct KTs; delete their own |
| `receiver` | The person taking over: open and export KTs |

Manage people from the server:

```
python -m auth add-user <tenant_id> alice@acme.com admin
python -m auth set-role <tenant_id> bob@acme.com reviewer
python -m auth disable-user <tenant_id> carol@acme.com     # ends her sessions immediately
python -m auth users <tenant_id>
python -m auth audit <tenant_id>                           # sign-ins, failures, role changes, deletions
```

## What is checked at sign-in

The ID token's signature against the provider's published keys (RSA or EC only), its issuer, audience, expiry and nonce; the PKCE verifier; that the sign-in finishes in the browser that started it; that the email domain belongs to the workspace and the email is not marked unverified. A person is identified by the provider's subject, not by their email. Sessions last 12 hours (`CONTINUUM_SESSION_TTL`); disabling a person or changing their role applies at their next request. Signing out of Continuum does not sign the person out of the identity provider.
