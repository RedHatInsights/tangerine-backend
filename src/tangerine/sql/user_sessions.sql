SELECT DISTINCT ON (session_uuid) *
FROM interactions
WHERE user_id = :user_id
ORDER BY session_uuid, timestamp ASC;
