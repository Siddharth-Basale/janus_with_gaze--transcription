import { config } from '../config/env.js';
import { JanusWsClient } from '../config/janusClient.js';
import { toRoomId } from '../utils/room.js';

const withClient = async (callback) => {
    const client = new JanusWsClient({});
    await client.connect();
    await client.createSession();
    await client.attachVideoRoom();

    try {
        const result = await callback(client);
        return result;
    } finally {
        await client.destroySession();
        await client.close();
    }
};

export const ensureRoom = async (roomName) =>
    withClient(async (client) => {
        const targetId = toRoomId(roomName);
        const rooms = await client.listRooms();
        const exists = rooms.find((room) => room.room === targetId);

        if (exists) {
            return { ...exists, roomName, roomId: targetId };
        }

        const creation = await client.createRoom({
            roomId: targetId,
            description: roomName,
            publishers: config.janusDefaultPublishers,
        });

        return { ...creation, roomName, roomId: targetId };
    });

export const listRooms = async () =>
    withClient(async (client) => {
        const rooms = await client.listRooms();
        return rooms.map((room) => ({
            roomId: room.room,
            description: room.description,
            maxPublishers: room.publishers,
        }));
    });

