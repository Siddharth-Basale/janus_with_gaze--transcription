import { Router } from 'express';
import { ensureRoom, listRooms } from '../services/roomService.js';

const router = Router();

router.get('/', async (req, res, next) => {
    try {
        const rooms = await listRooms();
        res.json({ rooms });
    } catch (err) {
        next(err);
    }
});

router.post('/', async (req, res, next) => {
    try {
        const { name } = req.body;
        if (!name) {
            return res.status(400).json({ error: 'Room name is required' });
        }
        const room = await ensureRoom(name);
        return res.status(201).json({ room });
    } catch (err) {
        return next(err);
    }
});

export default router;

