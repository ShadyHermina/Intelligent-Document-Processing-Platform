from core.config import get_settings

s = get_settings()

print('instance_id  :', s.instance_id)
print('postgres_host:', s.postgres_host)
print('postgres_port:', s.postgres_port, type(s.postgres_port))
print('postgres_db  :', s.postgres_db)
print('postgres_user:', s.postgres_user)
print('qdrant_host  :', s.qdrant_host)
print('qdrant_port  :', s.qdrant_port, type(s.qdrant_port))
print('app_env      :', s.app_env)
print('session_ttl  :', s.session_ttl_hours, 'hours')
print('openai_key   :', s.openai_api_key[:8], '...')
print('OK')