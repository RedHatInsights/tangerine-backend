SELECT  *
FROM interactions
WHERE user_id = :user_id AND session_uuid = :session_uuid
ORDER BY timestamp ASC;