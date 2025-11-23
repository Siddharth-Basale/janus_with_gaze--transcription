import crypto from 'node:crypto';

export const toRoomId = (roomName) => {
    const normalized = roomName.trim().toLowerCase();
    if (!normalized) {
        throw new Error('Room name is required');
    }
    const hash = crypto.createHash('sha1').update(normalized).digest();
    const value = hash.readUInt32BE(0);
    // Ensure Janus room ids stay within safe integer range
    return (value % 900000000) + 100000000;
};

